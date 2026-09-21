"""Maintenance work orders from the Limble CMMS."""
import math
import random
from datetime import date, datetime, timedelta

import pandas as pd

from ..config import (
    RANDOM_SEED, START_DATE, END_DATE, MACHINE_IDS, AGING_ASSETS,
    OEE_TARGET_BY_MACHINE, PM_INTERVAL_DAYS, PM_ONTIME_RATE,
    PM_ONTIME_RATE_AGING, PM_ADHOC_RATE, PM_OVERDUE_THRESHOLD_DAYS,
    MIN_OVERDUE_PMS_PER_MACHINE, FAILURE_CODES, FAILURE_CODE_WEIGHTS,
    EMPLOYEE_NUMBER_BY_OPERATOR, MAINT_TECH_IDS, AGING_ASSET_FAILURE_MULTIPLIER,
)

# Baseline unplanned repairs per machine per year, before health adjustment.
BASE_UNPLANNED_PER_YEAR = 6.0

# Unplanned failures follow a wear-out (increasing-hazard) renewal process rather
# than random placement, so time since the last failure predicts time to the
# next one. Shape > 1 is wear-out; larger is more predictable. While a machine is
# past its PM due date the hazard rises, so the gap to the next failure shrinks.
WEIBULL_SHAPE            = 4.5
PM_OVERDUE_HAZARD_FACTOR = 0.50


def _in_any(windows: list, d: date) -> bool:
    return any(start <= d <= end for start, end in windows)

PARTS_BY_CODE = {
    "TOOLING":          ["Insert set", "Tool holder", "Collet"],
    "MECHANICAL":       ["Spindle bearing", "Way cover", "Drive belt", "Ball screw"],
    "ELECTRICAL":       ["Servo amplifier", "Encoder", "Contactor", "Control fuse"],
    "OPERATOR_INDUCED": ["Fixture clamp", "Probe stylus"],
    "ENVIRONMENTAL":    ["Coolant filter", "Air filter", "Seal kit"],
    "PM":               ["Coolant filter", "Way lube", "Air filter"],
    "INSPECTION":       ["Inspection consumables"],
}

RESOLUTION_NOTES = {
    "TOOLING":          "Replaced worn tooling and confirmed first-article dimensions within tolerance.",
    "MECHANICAL":       "Serviced mechanical assembly, verified alignment and runout on completion.",
    "ELECTRICAL":       "Repaired electrical fault, tested control response through full axis travel.",
    "OPERATOR_INDUCED": "Corrected setup fault, reviewed procedure with the assigned operator.",
    "ENVIRONMENTAL":    "Cleared coolant and filtration issue, restored fluid levels to spec.",
    "PLANNED_PM":       "Completed scheduled preventive maintenance per equipment checklist.",
    "INSPECTION":       "Completed condition inspection, no corrective action required.",
}


def _tech_empids():
    return [EMPLOYEE_NUMBER_BY_OPERATOR[o] for o in MAINT_TECH_IDS]


def _parts_field(items: list) -> str:
    return "; ".join(items)


def generate_maintenance_records(machines_df: pd.DataFrame,
                                 operators_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    urng = random.Random(RANDOM_SEED + 2718)   # dedicated stream for failure timing
    tech_empids = _tech_empids()
    total_days  = (END_DATE - START_DATE).days

    records = []
    counter = 1

    def next_id():
        nonlocal counter
        mid = f"MNT-{counter:06d}"
        counter += 1
        return mid

    for machine_id in MACHINE_IDS:
        is_aging   = machine_id in AGING_ASSETS
        ontime_rate = PM_ONTIME_RATE_AGING if is_aging else PM_ONTIME_RATE
        overdue_windows = []   # date ranges when this machine is >14 days past a PM
        pm_records = []        # this machine's PM rows, for the overdue-coverage floor

        # ── Scheduled preventive maintenance ────────────────────────────────
        offset = rng.randint(0, PM_INTERVAL_DAYS - 1)
        scheduled = START_DATE + timedelta(days=offset)
        while scheduled <= END_DATE:
            adhoc = rng.random() < PM_ADHOC_RATE
            if adhoc:
                completed   = scheduled + timedelta(days=rng.randint(0, 20))
                sched_field = None
                overdue     = None
            elif rng.random() < ontime_rate:
                overdue     = rng.randint(-5, 0)   # completed on or before schedule
                completed   = scheduled + timedelta(days=overdue)
                sched_field = scheduled.isoformat()
            else:
                overdue     = rng.randint(1, 30)   # completed late
                completed   = scheduled + timedelta(days=overdue)
                sched_field = scheduled.isoformat()
                if overdue > 14:
                    overdue_windows.append((scheduled + timedelta(days=14), completed))

            if completed > END_DATE:
                scheduled += timedelta(days=PM_INTERVAL_DAYS)
                continue

            downtime = round(rng.uniform(2.0, 6.0), 1)
            open_dt  = datetime(completed.year, completed.month, completed.day, 7, 0)
            close_dt = open_dt + timedelta(hours=downtime)
            pm_records.append({
                "maintenance_id":       next_id(),
                "machine_id":           machine_id,
                "maintenance_type":     "PLANNED_PM",
                "failure_code":         None,
                "work_order_open_date": open_dt.isoformat(sep=" "),
                "work_order_close_date": close_dt.isoformat(sep=" "),
                "downtime_hours":       downtime,
                "technician_empid":     rng.choice(tech_empids),
                "parts_consumed":       _parts_field(
                    rng.sample(PARTS_BY_CODE["PM"], k=rng.randint(1, 2))),
                "resolution_notes":     RESOLUTION_NOTES["PLANNED_PM"],
                "pm_scheduled_date":    sched_field,
                "pm_completed_date":    completed.isoformat(),
                "days_overdue":         overdue,
            })
            scheduled += timedelta(days=PM_INTERVAL_DAYS)

        # Overdue-coverage floor: if this machine drew too few materially overdue
        # PMs, push a handful of on-time completions past the threshold so the
        # within-machine PM-overdue vs alarm-rate effect is observable everywhere.
        shortfall = MIN_OVERDUE_PMS_PER_MACHINE - len(overdue_windows)
        if shortfall > 0:
            for rec in pm_records:
                if shortfall <= 0:
                    break
                if rec["pm_scheduled_date"] is None or rec["days_overdue"] is None \
                        or rec["days_overdue"] > PM_OVERDUE_THRESHOLD_DAYS:
                    continue
                sched = date.fromisoformat(rec["pm_scheduled_date"])
                new_overdue = rng.randint(PM_OVERDUE_THRESHOLD_DAYS + 4,
                                          PM_OVERDUE_THRESHOLD_DAYS + 16)
                completed = sched + timedelta(days=new_overdue)
                if completed > END_DATE:
                    continue
                open_dt  = datetime(completed.year, completed.month, completed.day, 7, 0)
                close_dt = open_dt + timedelta(hours=rec["downtime_hours"])
                rec["days_overdue"]          = new_overdue
                rec["pm_completed_date"]      = completed.isoformat()
                rec["work_order_open_date"]   = open_dt.isoformat(sep=" ")
                rec["work_order_close_date"]  = close_dt.isoformat(sep=" ")
                overdue_windows.append((sched + timedelta(days=PM_OVERDUE_THRESHOLD_DAYS), completed))
                shortfall -= 1

        records.extend(pm_records)

        # ── Unplanned repairs: wear-out renewal process ─────────────────────
        # Increasing-hazard (Weibull) gaps between failures, so days since the last
        # failure predicts days to the next. The failure rate (and thus repair
        # count and downtime totals) matches the health model; only the timing is
        # now structured. The gap shrinks while the machine is past its PM due date.
        health_factor = (0.90 - OEE_TARGET_BY_MACHINE[machine_id]) * 4 + 1.0
        if is_aging:
            health_factor *= AGING_ASSET_FAILURE_MULTIPLIER
        per_year = BASE_UNPLANNED_PER_YEAR * health_factor
        mean_gap = 365.0 / per_year
        scale    = mean_gap / math.gamma(1 + 1 / WEIBULL_SHAPE)

        t = START_DATE + timedelta(days=int(urng.uniform(0, mean_gap)))
        while t <= END_DATE:
            event_day = t
            code      = urng.choices(FAILURE_CODES, weights=FAILURE_CODE_WEIGHTS)[0]
            downtime  = round(urng.uniform(2.0, 24.0) * (1.4 if is_aging else 1.0), 1)
            open_dt   = datetime(event_day.year, event_day.month, event_day.day,
                                 urng.randint(6, 20), urng.choice([0, 15, 30, 45]))
            close_dt  = open_dt + timedelta(hours=downtime)
            records.append({
                "maintenance_id":       next_id(),
                "machine_id":           machine_id,
                "maintenance_type":     "UNPLANNED_REPAIR",
                "failure_code":         code,
                "work_order_open_date": open_dt.isoformat(sep=" "),
                "work_order_close_date": close_dt.isoformat(sep=" "),
                "downtime_hours":       downtime,
                "technician_empid":     urng.choice(tech_empids),
                "parts_consumed":       _parts_field(
                    urng.sample(PARTS_BY_CODE[code],
                                k=min(len(PARTS_BY_CODE[code]), urng.randint(1, 3)))),
                "resolution_notes":     RESOLUTION_NOTES[code],
                "pm_scheduled_date":    None,
                "pm_completed_date":    None,
                "days_overdue":         None,
            })
            gap = urng.weibullvariate(scale, WEIBULL_SHAPE)
            if _in_any(overdue_windows, event_day):
                gap *= PM_OVERDUE_HAZARD_FACTOR
            t = t + timedelta(days=max(2, int(round(gap))))

        # ── Periodic condition inspections ──────────────────────────────────
        inspect = START_DATE + timedelta(days=rng.randint(0, 30))
        while inspect <= END_DATE:
            downtime = round(rng.uniform(0.5, 2.0), 1)
            open_dt  = datetime(inspect.year, inspect.month, inspect.day, 15, 0)
            close_dt = open_dt + timedelta(hours=downtime)
            records.append({
                "maintenance_id":       next_id(),
                "machine_id":           machine_id,
                "maintenance_type":     "INSPECTION",
                "failure_code":         None,
                "work_order_open_date": open_dt.isoformat(sep=" "),
                "work_order_close_date": close_dt.isoformat(sep=" "),
                "downtime_hours":       downtime,
                "technician_empid":     rng.choice(tech_empids),
                "parts_consumed":       _parts_field(PARTS_BY_CODE["INSPECTION"]),
                "resolution_notes":     RESOLUTION_NOTES["INSPECTION"],
                "pm_scheduled_date":    None,
                "pm_completed_date":    None,
                "days_overdue":         None,
            })
            inspect += timedelta(days=30)

    df = pd.DataFrame(records).sort_values(
        ["work_order_open_date", "machine_id"]).reset_index(drop=True)
    return df
