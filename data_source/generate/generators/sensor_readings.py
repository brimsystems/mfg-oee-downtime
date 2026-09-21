"""Condition-monitoring sensor readings from a retrofit IIoT gateway.

One daily summary reading per machine for four channels: spindle vibration,
bearing temperature, spindle motor power draw, and hydraulic pressure. Readings
sit at a machine-specific baseline with normal variation, then drift and spike
in the days before an unplanned failure. The pre-failure signature is
mode-specific (mechanical shows in vibration and temperature, tooling in
vibration and power, electrical in power and temperature, environmental in
hydraulic pressure). Operator-induced failures carry no sensor precursor, which
gives the RUL model a realistic accuracy ceiling. Failure dates are read from
the CMMS maintenance records so the sensors lead the same failures the target
is built from.
"""
import random
from datetime import datetime

import pandas as pd

from ..config import RANDOM_SEED, MACHINES_DATA, AGE_BY_MACHINE, operating_days

TYPE_BY_MACHINE = {m[0]: m[1] for m in MACHINES_DATA}

# Baseline reading by machine type: vibration (mm/s RMS), bearing temp (C),
# spindle power (kW), hydraulic pressure (bar).
SENSOR_BASE = {
    "CNC Lathe":       {"vib": 1.1, "temp": 42.0, "power": 5.0,  "press": 118.0},
    "Vertical Mill":   {"vib": 1.5, "temp": 46.0, "power": 7.5,  "press": 125.0},
    "Horizontal Mill": {"vib": 1.9, "temp": 50.0, "power": 11.0, "press": 130.0},
}
# Normal day-to-day variation (standard deviation). Kept tight so the
# pre-failure signature stands clear of baseline noise.
NOISE = {"vib": 0.10, "temp": 1.1, "power": 0.30, "press": 2.2}
# Per-year-of-age baseline shift (older machines run hotter, rougher, harder).
AGE_COEF = {"vib": 0.045, "temp": 0.55, "power": 0.10, "press": -0.35}

# Nominal failure-mode signature: which channels CAN react, and the full-strength
# peak of each. Multiplicative peak factor for vib/power, additive delta (C) for
# temp, multiplicative factor for press (a drop when < 1). The realised behaviour
# is deliberately noisy (below) so the model is not unrealistically effective.
MODE_SIGNATURE = {
    "MECHANICAL":       {"vib": 3.2, "temp": 26.0},
    "TOOLING":          {"vib": 2.6, "power": 1.80},
    "ELECTRICAL":       {"power": 1.9, "temp": 20.0},
    "ENVIRONMENTAL":    {"press": 0.58, "temp": 10.0},
    "OPERATOR_INDUCED": {},   # no condition-monitoring precursor
}

# Stochastic realism: not every failure telegraphs itself, not every eligible
# channel reacts, and the magnitude and lead time vary event to event. Combined
# with the operator-induced mode (no precursor), this leaves a meaningful share
# of failures unforeseeable and produces occasional false positives, so the model
# lands at a realistic rather than perfect accuracy.
PRECURSOR_PROB     = 0.90          # chance a non-operator failure has any precursor
CHANNEL_PROB       = 0.85          # chance each eligible channel actually reacts
MAG_MIN, MAG_MAX   = 0.80, 1.40    # per-event magnitude scale on the nominal peak
RAMP_MIN, RAMP_MAX = 14, 28        # per-event lead time (days)
BENIGN_SPIKE_PROB  = 0.015         # isolated anomaly day unrelated to any failure
BENIGN = {"vib": 1.6, "temp": 9.0, "power": 1.35, "press": 0.82}


def _failures_by_machine(maintenance_df: pd.DataFrame) -> dict:
    unpl = maintenance_df[maintenance_df["maintenance_type"] == "UNPLANNED_REPAIR"]
    out = {}
    for _, r in unpl.iterrows():
        d = datetime.fromisoformat(str(r["work_order_open_date"])).date()
        out.setdefault(r["machine_id"], []).append((d, r["failure_code"]))
    for mid in out:
        out[mid].sort()
    return out


def generate_sensor_readings(machines_df: pd.DataFrame,
                             maintenance_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED + 5150)
    failures = _failures_by_machine(maintenance_df)
    days = operating_days()

    cols = {c: [] for c in ("reading_date", "machine_id", "vibration_rms_mm_s",
                            "bearing_temp_c", "spindle_power_kw", "hydraulic_pressure_bar")}

    for machine_id in TYPE_BY_MACHINE:
        mtype = TYPE_BY_MACHINE[machine_id]
        age   = AGE_BY_MACHINE[machine_id]
        base  = SENSOR_BASE[mtype]
        b_vib   = base["vib"]   + AGE_COEF["vib"]   * age
        b_temp  = base["temp"]  + AGE_COEF["temp"]  * age
        b_power = base["power"] + AGE_COEF["power"] * age
        b_press = base["press"] + AGE_COEF["press"] * age

        # Precompute this machine's precursor windows: a random majority of its
        # failures, each with a random channel subset, magnitude, and lead time.
        precursors = []
        for fdate, code in failures.get(machine_id, []):
            sig = MODE_SIGNATURE.get(code, {})
            if not sig or rng.random() >= PRECURSOR_PROB:
                continue
            chans = [c for c in sig if rng.random() < CHANNEL_PROB] or [rng.choice(list(sig))]
            precursors.append({
                "date": fdate,
                "ramp": rng.randint(RAMP_MIN, RAMP_MAX),
                "sig": {c: sig[c] for c in chans},
                "strength": rng.uniform(MAG_MIN, MAG_MAX),
            })

        for day in days:
            vib   = b_vib   + rng.gauss(0, NOISE["vib"])
            temp  = b_temp  + rng.gauss(0, NOISE["temp"])
            power = b_power + rng.gauss(0, NOISE["power"])
            press = b_press + rng.gauss(0, NOISE["press"])

            pc, prog = None, 0.0
            for p in precursors:
                gap = (p["date"] - day).days
                if 0 <= gap <= p["ramp"]:
                    pc, prog = p, 1.0 - gap / p["ramp"]
                    break

            if pc:
                s, sig = pc["strength"] * prog, pc["sig"]
                if "vib" in sig:
                    vib   = b_vib   * (1 + (sig["vib"] - 1) * s) + rng.gauss(0, NOISE["vib"])
                if "temp" in sig:
                    temp  = b_temp  + sig["temp"] * s + rng.gauss(0, NOISE["temp"])
                if "power" in sig:
                    power = b_power * (1 + (sig["power"] - 1) * s) + rng.gauss(0, NOISE["power"])
                if "press" in sig:
                    press = b_press * (1 - (1 - sig["press"]) * s) + rng.gauss(0, NOISE["press"])
            elif rng.random() < BENIGN_SPIKE_PROB:
                ch = rng.choice(["vib", "temp", "power", "press"])
                if ch == "vib":
                    vib   = b_vib   * BENIGN["vib"]   + rng.gauss(0, NOISE["vib"])
                elif ch == "temp":
                    temp  = b_temp  + BENIGN["temp"]  + rng.gauss(0, NOISE["temp"])
                elif ch == "power":
                    power = b_power * BENIGN["power"] + rng.gauss(0, NOISE["power"])
                else:
                    press = b_press * BENIGN["press"] + rng.gauss(0, NOISE["press"])

            cols["reading_date"].append(day.isoformat())
            cols["machine_id"].append(machine_id)
            cols["vibration_rms_mm_s"].append(round(max(0.1, vib), 2))
            cols["bearing_temp_c"].append(round(temp, 1))
            cols["spindle_power_kw"].append(round(max(0.5, power), 2))
            cols["hydraulic_pressure_bar"].append(round(max(0.0, press), 1))

    return pd.DataFrame(cols)
