"""Job orders from the JobBOSS2 ERP."""
import random
from datetime import datetime, timedelta

import pandas as pd

from ..config import (
    RANDOM_SEED, MACHINE_IDS, SHIFT_HOURS, MATERIAL_TYPES, MATERIAL_WEIGHTS,
    CUSTOMERS, CUSTOMER_WEIGHTS, JOB_STATUSES, EMPLOYEE_NUMBER_BY_OPERATOR,
    EXTENDED_SETUP_OPERATORS, EXTENDED_SETUP_RATIO_RANGE, SETUP_HOURS_MEDIAN,
    SETUP_HOURS_JOB_STD, SETUP_OPERATOR_SPREAD_STD, SETUP_OPERATOR_SPREAD_CLIP,
    WORK_ORDERS_PER_DAY_MIN, WORK_ORDERS_PER_DAY_MAX, END_DATE, operating_days,
)

N_PARTS = 40
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]
COMPLEXITY_WEIGHTS = [0.45, 0.40, 0.15]

# Jobs opened within this many days of the window end may still be running.
OPEN_JOB_HORIZON_DAYS = 21


def _build_parts(rng: random.Random) -> list:
    parts = []
    for _ in range(N_PARTS):
        part_number = f"{rng.randint(10000, 99999)}-{rng.randint(1, 12):02d}"
        material    = rng.choices(MATERIAL_TYPES, weights=MATERIAL_WEIGHTS)[0]
        complexity  = rng.choices(COMPLEXITY_LEVELS, weights=COMPLEXITY_WEIGHTS)[0]
        base_hours  = {"Low": rng.uniform(1.0, 3.0),
                       "Medium": rng.uniform(3.0, 6.0),
                       "High": rng.uniform(6.0, 12.0)}[complexity]
        parts.append({
            "part_number":    part_number,
            "material_type":  material,
            "complexity":     complexity,
            "scheduled_hours": round(base_hours, 1),
        })
    return parts


def generate_work_orders(machines_df: pd.DataFrame,
                         operators_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    parts = _build_parts(rng)

    machinists_by_shift = {"A": [], "B": []}
    ops = operators_df[operators_df["role"] == "Machinist"]
    for _, row in ops.iterrows():
        shift = "A" if row["shift"] == "Shift A" else "B"
        machinists_by_shift[shift].append(int(row["employee_number"]))

    # Each machinist works to their own habitual setup level: most cluster close
    # to the cohort median, while the three flagged operators consistently run
    # longer. Drawn on a dedicated stream so the level assignment does not shift
    # the rest of the job sequence.
    setup_rng = random.Random(RANDOM_SEED + 4242)
    extended_setup_empids = {EMPLOYEE_NUMBER_BY_OPERATOR[o]
                             for o in EXTENDED_SETUP_OPERATORS}
    lo, hi = SETUP_OPERATOR_SPREAD_CLIP
    setup_factor_by_empid = {}
    for empid in sorted(EMPLOYEE_NUMBER_BY_OPERATOR.values()):
        if empid in extended_setup_empids:
            setup_factor_by_empid[empid] = setup_rng.uniform(*EXTENDED_SETUP_RATIO_RANGE)
        else:
            setup_factor_by_empid[empid] = min(
                max(setup_rng.gauss(1.0, SETUP_OPERATOR_SPREAD_STD), lo), hi)

    open_threshold = END_DATE - timedelta(days=OPEN_JOB_HORIZON_DAYS)

    records = []
    counter = 100000
    for day in operating_days():
        for _ in range(rng.randint(WORK_ORDERS_PER_DAY_MIN, WORK_ORDERS_PER_DAY_MAX)):
            machine_id = rng.choice(MACHINE_IDS)
            shift      = rng.choice(["A", "B"])
            pool       = machinists_by_shift[shift]
            if not pool:
                continue
            operator_empid = rng.choice(pool)
            part = rng.choice(parts)

            complexity_factor = {"Low": 1.0, "Medium": 1.15, "High": 1.4}[part["complexity"]]
            scheduled_hours = part["scheduled_hours"]
            actual_hours = round(scheduled_hours
                                 * rng.uniform(0.9, 1.1)
                                 * (complexity_factor if rng.random() < 0.5 else 1.0), 1)

            operator_setup_level = SETUP_HOURS_MEDIAN * setup_factor_by_empid.get(operator_empid, 1.0)
            setup_hours = round(max(0.1, rng.gauss(operator_setup_level, SETUP_HOURS_JOB_STD)), 2)

            start_h = SHIFT_HOURS[shift][0]
            scheduled_start = datetime(day.year, day.month, day.day,
                                       start_h + rng.randint(0, 1),
                                       rng.choice([0, 15, 30, 45]))
            scheduled_end = scheduled_start + timedelta(hours=scheduled_hours)
            actual_start  = scheduled_start + timedelta(minutes=rng.randint(-30, 60))
            actual_end    = actual_start + timedelta(hours=actual_hours + setup_hours)

            if day >= open_threshold:
                status = rng.choices(JOB_STATUSES, weights=[0.4, 0.4, 0.2])[0]
            else:
                status = rng.choices(JOB_STATUSES, weights=[0.94, 0.0, 0.06])[0]

            complete = status == "COMPLETE"
            records.append({
                "work_order_id":     f"WO-{counter:06d}",
                "machine_id":        machine_id,
                "operator_empid":    operator_empid,
                "part_number":       part["part_number"],
                "customer_id":       rng.choices(CUSTOMERS, weights=CUSTOMER_WEIGHTS)[0],
                "scheduled_start":   scheduled_start.isoformat(sep=" "),
                "scheduled_end":     scheduled_end.isoformat(sep=" "),
                "actual_start":      actual_start.isoformat(sep=" "),
                "actual_end":        actual_end.isoformat(sep=" ") if complete else None,
                "scheduled_hours":   scheduled_hours,
                "actual_hours":      actual_hours if complete else None,
                "setup_hours_actual": setup_hours,
                "job_status":        status,
                "material_type":     part["material_type"],
            })
            counter += 1

    return pd.DataFrame(records)
