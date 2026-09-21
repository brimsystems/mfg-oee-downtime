"""Machine state log from MachineMetrics (MTConnect)."""
import random
from datetime import datetime, timedelta

import pandas as pd

from ..config import (
    RANDOM_SEED, MACHINE_IDS, SHIFT_HOURS, STATE_INTERVAL_MINUTES,
    TIER_TIME_SHARES, UNPLANNED_DOWN_INTERVAL_RANGE, PLANNED_DOWN_INTERVAL_RANGE,
    SPINDLE_UTILIZATION_BASE, SPINDLE_UTILIZATION_AGE_SLOPE,
    SPINDLE_UTILIZATION_OFFSET_STD, SPINDLE_UTILIZATION_FLOOR,
    SPINDLE_UTILIZATION_STD, SPINDLE_UTILIZATION_DAILY_STD, AGE_BY_MACHINE,
    ALARM_CODES, AGING_ASSETS,
    AGING_ASSET_FAILURE_MULTIPLIER, AGING_ASSET_ALARM_MULTIPLIER,
    SHIFT_STARTUP_WINDOW_MINUTES,
    SHIFT_STARTUP_DOWNTIME_MULTIPLIER, PM_OVERDUE_ALARM_MULTIPLIER,
    PM_OVERDUE_THRESHOLD_DAYS, machine_tier, operating_days,
)

# Mean event duration in minutes, used to convert time shares into per-interval
# (per-decision) probabilities: a state's decision weight is its time share
# divided by how long each occurrence lasts.
_MEAN_INTERVALS = {
    "UNPLANNED_DOWN": sum(UNPLANNED_DOWN_INTERVAL_RANGE) / 2,
    "PLANNED_DOWN":   sum(PLANNED_DOWN_INTERVAL_RANGE) / 2,
}
MEAN_DURATION = {
    "RUNNING": 1, "IDLE": 1, "SETUP": 1, "ALARM": 1,
    "UNPLANNED_DOWN": _MEAN_INTERVALS["UNPLANNED_DOWN"],
    "PLANNED_DOWN":   _MEAN_INTERVALS["PLANNED_DOWN"],
}


def _decision_probs(tier: str) -> dict:
    """Convert a tier's time shares into per-interval decision probabilities."""
    shares = TIER_TIME_SHARES[tier]
    weights = {s: shares[s] / MEAN_DURATION[s] for s in shares}
    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}


def _fleet_mean(state: str) -> float:
    return sum(_decision_probs(machine_tier(m))[state]
               for m in MACHINE_IDS) / len(MACHINE_IDS)


def _machine_params() -> dict:
    """Per-interval state probabilities and spindle mean per machine."""
    aging_unplanned = AGING_ASSET_FAILURE_MULTIPLIER * _fleet_mean("UNPLANNED_DOWN")
    # Alarm elevation on aging assets is set against the mid-tier peer level, not
    # the fleet mean, so the aging machines read as modestly noisier than their
    # healthier peers rather than several times worse.
    aging_alarm     = AGING_ASSET_ALARM_MULTIPLIER * _decision_probs("mid")["ALARM"]

    # Fixed per-machine spindle offset so the age-performance relationship
    # carries realistic scatter rather than a perfectly linear fit.
    offset_rng = random.Random(RANDOM_SEED)

    params = {}
    for machine_id in MACHINE_IDS:
        tier  = machine_tier(machine_id)
        probs = _decision_probs(tier)
        if machine_id in AGING_ASSETS:
            probs["UNPLANNED_DOWN"] = aging_unplanned
            probs["ALARM"]          = aging_alarm
            fixed = (probs["IDLE"] + probs["SETUP"] + probs["PLANNED_DOWN"]
                     + probs["UNPLANNED_DOWN"] + probs["ALARM"])
            probs["RUNNING"] = max(0.05, 1.0 - fixed)
        offset = offset_rng.gauss(0, SPINDLE_UTILIZATION_OFFSET_STD)
        spindle_mean = max(
            SPINDLE_UTILIZATION_FLOOR,
            SPINDLE_UTILIZATION_BASE
            - SPINDLE_UTILIZATION_AGE_SLOPE * AGE_BY_MACHINE[machine_id]
            + offset,
        )
        params[machine_id] = {"probs": probs, "spindle_mean": spindle_mean}
    return params


def _build_overdue_windows(maintenance_df: pd.DataFrame) -> dict:
    """Date ranges during which a machine is more than the threshold past a due PM."""
    windows = {m: [] for m in MACHINE_IDS}
    pm = maintenance_df[
        (maintenance_df["maintenance_type"] == "PLANNED_PM")
        & maintenance_df["pm_scheduled_date"].notna()
        & maintenance_df["days_overdue"].notna()
    ]
    for _, row in pm.iterrows():
        overdue = row["days_overdue"]
        if overdue is None or overdue <= PM_OVERDUE_THRESHOLD_DAYS:
            continue
        scheduled = datetime.fromisoformat(row["pm_scheduled_date"]).date()
        completed = datetime.fromisoformat(row["pm_completed_date"]).date()
        start = scheduled + timedelta(days=PM_OVERDUE_THRESHOLD_DAYS)
        windows[row["machine_id"]].append((start, completed))
    return windows


def _in_overdue_window(windows: list, day) -> bool:
    return any(start <= day < end for start, end in windows)


def _draw_state(probs: dict, unplanned_mult: float, alarm_mult: float,
                rng: random.Random) -> str:
    p_unpl    = min(probs["UNPLANNED_DOWN"] * unplanned_mult, 0.60)
    p_alarm   = min(probs["ALARM"] * alarm_mult, 0.40)
    p_planned = probs["PLANNED_DOWN"]
    fixed     = p_unpl + p_alarm + p_planned
    productive = probs["RUNNING"] + probs["IDLE"] + probs["SETUP"]
    scale = (1.0 - fixed) / productive if productive > 0 else 0.0

    cumulative = [
        ("RUNNING",        probs["RUNNING"] * scale),
        ("IDLE",           probs["IDLE"] * scale),
        ("SETUP",          probs["SETUP"] * scale),
        ("PLANNED_DOWN",   p_planned),
        ("UNPLANNED_DOWN", p_unpl),
        ("ALARM",          p_alarm),
    ]
    r, acc = rng.random(), 0.0
    for state, p in cumulative:
        acc += p
        if r <= acc:
            return state
    return "RUNNING"


def generate_production_events(machines_df: pd.DataFrame,
                               operators_df: pd.DataFrame,
                               maintenance_df: pd.DataFrame) -> pd.DataFrame:
    rng     = random.Random(RANDOM_SEED)
    # Dedicated stream for the per-day performance offset so it moves daily and
    # weekly Performance without perturbing the main state-draw sequence (keeping
    # Availability and downtime totals stable).
    perf_rng = random.Random(RANDOM_SEED + 90210)
    params  = _machine_params()
    windows = _build_overdue_windows(maintenance_df)

    machinists_by_shift = {"A": [], "B": []}
    ops = operators_df[operators_df["role"] == "Machinist"]
    for _, row in ops.iterrows():
        shift = "A" if row["shift"] == "Shift A" else "B"
        machinists_by_shift[shift].append(int(row["employee_number"]))

    interval = timedelta(minutes=STATE_INTERVAL_MINUTES)
    cols = {c: [] for c in (
        "event_id", "machine_id", "event_timestamp", "shift", "machine_state",
        "state_duration_minutes", "operator_id", "spindle_utilization_pct",
        "alarm_code")}
    counter = 1

    for day in operating_days():
        # One zero-mean performance offset per machine per day, shared across both
        # shifts, so a single day reads differently from a weekly average.
        perf_offset = {m: perf_rng.gauss(0, SPINDLE_UTILIZATION_DAILY_STD)
                       for m in MACHINE_IDS}
        for shift, (start_h, end_h) in SHIFT_HOURS.items():
            pool = machinists_by_shift[shift]
            shift_start = datetime(day.year, day.month, day.day, start_h, 0)
            shift_end   = datetime(day.year, day.month, day.day, end_h, 0)
            overdue_today = {
                m: _in_overdue_window(windows[m], day) for m in MACHINE_IDS
            }
            for machine_id in MACHINE_IDS:
                p = params[machine_id]
                operator_empid = rng.choice(pool) if pool else None
                t = shift_start
                while t < shift_end:
                    minute_into_shift = (t.hour * 60 + t.minute) - start_h * 60
                    unpl_mult = (SHIFT_STARTUP_DOWNTIME_MULTIPLIER[shift]
                                 if minute_into_shift < SHIFT_STARTUP_WINDOW_MINUTES
                                 else 1.0)
                    alarm_mult = (PM_OVERDUE_ALARM_MULTIPLIER
                                  if overdue_today[machine_id] else 1.0)

                    state = _draw_state(p["probs"], unpl_mult, alarm_mult, rng)

                    if state == "UNPLANNED_DOWN":
                        span = STATE_INTERVAL_MINUTES * rng.randint(*UNPLANNED_DOWN_INTERVAL_RANGE)
                    elif state == "PLANNED_DOWN":
                        span = STATE_INTERVAL_MINUTES * rng.randint(*PLANNED_DOWN_INTERVAL_RANGE)
                    else:
                        span = STATE_INTERVAL_MINUTES
                    remaining = int((shift_end - t).total_seconds() // 60)
                    duration = min(span, remaining)

                    spindle = alarm = None
                    if state == "RUNNING":
                        spindle = round(min(100.0, max(40.0, rng.gauss(
                            p["spindle_mean"] + perf_offset[machine_id],
                            SPINDLE_UTILIZATION_STD))), 1)
                    elif state == "ALARM":
                        alarm = rng.choice(ALARM_CODES)

                    cols["event_id"].append(f"EVT-{counter:08d}")
                    cols["machine_id"].append(machine_id)
                    cols["event_timestamp"].append(t.isoformat(sep=" "))
                    cols["shift"].append(shift)
                    cols["machine_state"].append(state)
                    cols["state_duration_minutes"].append(duration)
                    cols["operator_id"].append(operator_empid)
                    cols["spindle_utilization_pct"].append(spindle)
                    cols["alarm_code"].append(alarm)
                    counter += 1

                    t += timedelta(minutes=duration) if duration else interval

    return pd.DataFrame(cols)
