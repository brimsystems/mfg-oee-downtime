"""Build all source-system extracts and write raw and sample files."""
from pathlib import Path

import pandas as pd

from .config import (
    RAW_DIR, SAMPLES_DIR, SAMPLE_SIZE, TABLE_SYSTEM_MAP, START_DATE, END_DATE,
    AGING_ASSETS, SHIFT_HOURS, SHIFT_STARTUP_WINDOW_MINUTES,
    EXTENDED_SETUP_OPERATORS, EMPLOYEE_NUMBER_BY_OPERATOR,
)
from .generators.machines            import generate_machines
from .generators.operators           import generate_operators
from .generators.maintenance_records import generate_maintenance_records
from .generators.production_events   import generate_production_events
from .generators.work_orders         import generate_work_orders
from .generators.sensor_readings     import generate_sensor_readings


def _save(df: pd.DataFrame, name: str, base_dir: Path, sample: bool = False) -> None:
    system   = TABLE_SYSTEM_MAP[name]
    out_dir  = base_dir / system
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix   = "_sample" if sample else ""
    filepath = out_dir / f"{name}{suffix}.csv"
    df.to_csv(filepath, index=False)
    size_kb = filepath.stat().st_size / 1024
    size    = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
    print(f"  [{system:>14}]  {name + suffix:<28} {len(df):>9,} rows   {size}")


def _rate(numer, denom):
    return numer / denom if denom else 0.0


def run() -> None:
    print(f"\nBuilding source extracts for {START_DATE} through {END_DATE}\n")

    print("[1/6] Machines            (MachineMetrics)")
    machines = generate_machines()

    print("[2/6] Operators           (ADP HR)")
    operators = generate_operators()

    print("[3/6] Maintenance records (Limble CMMS)")
    maintenance = generate_maintenance_records(machines, operators)

    print("[4/6] Production events   (MachineMetrics) — this step takes a moment")
    production = generate_production_events(machines, operators, maintenance)

    print("[5/6] Work orders         (JobBOSS2 ERP)")
    work_orders = generate_work_orders(machines, operators)

    print("[6/6] Sensor readings     (IIoT condition monitoring)")
    sensors = generate_sensor_readings(machines, maintenance)

    tables = {
        "machines":            machines,
        "operators":           operators,
        "production_events":   production,
        "work_orders":         work_orders,
        "maintenance_records": maintenance,
        "sensor_readings":     sensors,
    }

    print(f"\nFull extracts -> {RAW_DIR}")
    for name, df in tables.items():
        _save(df, name, RAW_DIR)

    print(f"\nSample extracts ({SAMPLE_SIZE} rows) -> {SAMPLES_DIR}")
    for name, df in tables.items():
        _save(df.head(SAMPLE_SIZE), name, SAMPLES_DIR, sample=True)

    _summary(machines, operators, production, work_orders, maintenance)


def _summary(machines, operators, production, work_orders, maintenance) -> None:
    print("\n" + "=" * 72)
    print("RELIABILITY & DATA SUMMARY")
    print("=" * 72)

    prod = production.copy()
    prod["down_or_alarm"] = prod["machine_state"].isin(["UNPLANNED_DOWN", "ALARM"])

    # Failure exposure of the oldest assets vs the rest of the fleet.
    aging = prod[prod["machine_id"].isin(AGING_ASSETS)]
    rest  = prod[~prod["machine_id"].isin(AGING_ASSETS)]
    aging_rate = _rate(aging["down_or_alarm"].sum(), len(aging))
    rest_rate  = _rate(rest["down_or_alarm"].sum(), len(rest))
    print(f"\nOldest assets ({', '.join(AGING_ASSETS)}) unplanned-down + alarm share")
    print(f"  oldest {aging_rate:.1%}   remaining fleet {rest_rate:.1%}"
          f"   ratio {_rate(aging_rate, rest_rate):.1f}x")

    # Shift-startup downtime concentration, by shift.
    ts       = pd.to_datetime(prod["event_timestamp"])
    shift_start_min = prod["shift"].map({"A": SHIFT_HOURS["A"][0] * 60,
                                         "B": SHIFT_HOURS["B"][0] * 60})
    into_shift = (ts.dt.hour * 60 + ts.dt.minute) - shift_start_min
    print(f"\nShift startup window (first {SHIFT_STARTUP_WINDOW_MINUTES} min) unplanned-down share")
    for sh in ["A", "B"]:
        sp = prod[(prod["shift"] == sh) & (into_shift < SHIFT_STARTUP_WINDOW_MINUTES)]
        rs = prod[(prod["shift"] == sh) & (into_shift >= SHIFT_STARTUP_WINDOW_MINUTES)]
        sr = _rate((sp["machine_state"] == "UNPLANNED_DOWN").sum(), len(sp))
        rr = _rate((rs["machine_state"] == "UNPLANNED_DOWN").sum(), len(rs))
        print(f"  Shift {sh}: startup {sr:.1%}  rest {rr:.1%}  ratio {_rate(sr, rr):.1f}x")

    # Preventive-maintenance compliance.
    pm = maintenance[(maintenance["maintenance_type"] == "PLANNED_PM")
                     & maintenance["days_overdue"].notna()]
    ontime = _rate((pm["days_overdue"] <= 0).sum(), len(pm))
    aging_pm = pm[pm["machine_id"].isin(AGING_ASSETS)]
    aging_ontime = _rate((aging_pm["days_overdue"] <= 0).sum(), len(aging_pm))
    print(f"\nPreventive-maintenance on-time completion")
    print(f"  fleet {ontime:.1%}   oldest assets {aging_ontime:.1%}")

    # Setup-time exposure for the flagged operators.
    ext_empids = {EMPLOYEE_NUMBER_BY_OPERATOR[o] for o in EXTENDED_SETUP_OPERATORS}
    flagged = work_orders[work_orders["operator_empid"].isin(ext_empids)]
    cohort  = work_orders[~work_orders["operator_empid"].isin(ext_empids)]
    flagged_med = flagged["setup_hours_actual"].median()
    cohort_med  = cohort["setup_hours_actual"].median()
    print(f"\nSetup hours — flagged operators vs cohort median")
    print(f"  flagged {flagged_med:.2f} h   cohort {cohort_med:.2f} h"
          f"   ratio {_rate(flagged_med, cohort_med):.1f}x")

    # Integration and data-quality notes.
    adhoc = maintenance[(maintenance["maintenance_type"] == "PLANNED_PM")
                        & maintenance["pm_scheduled_date"].isna()]
    open_jobs = work_orders["actual_end"].isna().mean()
    print(f"\nData notes")
    print(f"  ad-hoc PMs with no scheduled date  {_rate(len(adhoc), len(maintenance[maintenance['maintenance_type']=='PLANNED_PM'])):.0%}")
    print(f"  open jobs with no actual end       {open_jobs:.0%}")
    print(f"  operator identifiers               HR OPR-XXX vs payroll number")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run()
