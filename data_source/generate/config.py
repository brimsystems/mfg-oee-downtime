"""Central configuration for the OEE and predictive-maintenance data platform."""
from datetime import date, timedelta
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
RAW_DIR     = REPO_ROOT / "data_source" / "raw"
SAMPLES_DIR = REPO_ROOT / "data_source" / "samples"

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 200

# ── Observation window ──────────────────────────────────────────────────────
# 36-month training/validation/test span plus a three-month forward window
# used for monthly scoring and monitoring.
START_DATE           = date(2023, 1, 2)   # first Monday of 2023
END_DATE             = date(2026, 3, 31)
SCORING_WINDOW_START = date(2026, 1, 1)
FLEET_REFERENCE_DATE = date(2026, 1, 1)   # anchor for deriving install dates

# ── Shift calendar ──────────────────────────────────────────────────────────
# Production runs two shifts, Monday through Saturday.
SHIFT_HOURS = {
    "A": (6, 14),
    "B": (14, 22),
}
STATE_INTERVAL_MINUTES = 15

# ── Source system → output subdirectory ─────────────────────────────────────
TABLE_SYSTEM_MAP = {
    "machines":            "machinemetrics",
    "operators":           "hr",
    "production_events":   "machinemetrics",
    "work_orders":         "erp",
    "maintenance_records": "cmms",
    "sensor_readings":     "sensors",
}

# ── Machine master ──────────────────────────────────────────────────────────
# (machine_id, machine_type, controller_type, location_cell, age_years, max_spindle_rpm)
MACHINES_DATA = [
    ("MCH-001", "CNC Lathe",       "Fanuc", "Cell A",  2,  5000),
    ("MCH-002", "CNC Lathe",       "Fanuc", "Cell A",  4,  4500),
    ("MCH-003", "CNC Lathe",       "Fanuc", "Cell A",  6,  5000),
    ("MCH-004", "CNC Lathe",       "Fanuc", "Cell A",  8,  4000),
    ("MCH-005", "Vertical Mill",   "Haas",  "Cell B",  2, 12000),
    ("MCH-006", "Vertical Mill",   "Haas",  "Cell B",  5, 10000),
    ("MCH-007", "Vertical Mill",   "Haas",  "Cell B", 12,  8000),
    ("MCH-008", "Vertical Mill",   "Haas",  "Cell B",  9, 10000),
    ("MCH-009", "Horizontal Mill", "Mazak", "Cell C",  7, 15000),
    ("MCH-010", "Horizontal Mill", "Mazak", "Cell C",  8, 12000),
    ("MCH-011", "Horizontal Mill", "Mazak", "Cell C", 13, 10000),
    ("MCH-012", "Horizontal Mill", "Mazak", "Cell C", 10, 12000),
]
MACHINE_IDS = [m[0] for m in MACHINES_DATA]
AGE_BY_MACHINE = {m[0]: m[4] for m in MACHINES_DATA}

# Composite OEE the state distributions are calibrated toward, by machine.
OEE_TARGET_BY_MACHINE = {
    "MCH-001": 0.82, "MCH-002": 0.82, "MCH-003": 0.82, "MCH-004": 0.82,
    "MCH-005": 0.74, "MCH-006": 0.74, "MCH-007": 0.74, "MCH-008": 0.74,
    "MCH-009": 0.74, "MCH-010": 0.74, "MCH-011": 0.61, "MCH-012": 0.61,
}

# Target share of scheduled time spent in each state, by reliability tier.
# These are TIME shares (not event frequencies) and directly drive OEE:
# Availability = RUNNING / (1 - PLANNED_DOWN). Event frequencies are derived
# from these shares and the event durations below. Columns sum to 1.0.
# Tier is assigned from the machine's OEE target.
STATE_TIERS = ("RUNNING", "IDLE", "SETUP", "UNPLANNED_DOWN", "PLANNED_DOWN", "ALARM")
TIER_TIME_SHARES = {
    "high": {"RUNNING": 0.851, "IDLE": 0.05, "SETUP": 0.04,
             "UNPLANNED_DOWN": 0.020, "PLANNED_DOWN": 0.03, "ALARM": 0.009},
    "mid":  {"RUNNING": 0.805, "IDLE": 0.06, "SETUP": 0.04,
             "UNPLANNED_DOWN": 0.040, "PLANNED_DOWN": 0.03, "ALARM": 0.025},
    "low":  {"RUNNING": 0.704, "IDLE": 0.10, "SETUP": 0.05,
             "UNPLANNED_DOWN": 0.080, "PLANNED_DOWN": 0.04, "ALARM": 0.026},
}

# Downtime event duration, in number of 15-minute intervals (min, max inclusive).
# Running/idle/setup/alarm occupy a single interval.
UNPLANNED_DOWN_INTERVAL_RANGE = (2, 10)   # mean ~90 minutes
PLANNED_DOWN_INTERVAL_RANGE   = (4, 12)   # mean ~120 minutes

# Spindle utilisation is the Performance component of OEE. It declines with
# machine age (worn spindles, older controls, more conservative feeds and
# speeds): mean = BASE - AGE_SLOPE * age_years, plus a fixed per-machine offset,
# floored. The per-machine offset carries the scatter that keeps the fleet
# age-performance fit realistic (about r = -0.8) rather than a near-perfect line.
# A separate per-day offset (PERF_DAILY_STD) gives each machine day-to-day
# movement so daily and weekly performance differ; it is zero-mean, so the
# long-run per-machine level is unchanged. Drawn per running interval with the
# interval standard deviation below.
SPINDLE_UTILIZATION_BASE       = 92.0
SPINDLE_UTILIZATION_AGE_SLOPE  = 1.4
SPINDLE_UTILIZATION_OFFSET_STD = 5.0
SPINDLE_UTILIZATION_FLOOR      = 68.0
SPINDLE_UTILIZATION_STD        = 5.0
SPINDLE_UTILIZATION_DAILY_STD  = 3.0

def machine_tier(machine_id: str) -> str:
    oee = OEE_TARGET_BY_MACHINE[machine_id]
    if oee >= 0.80:
        return "high"
    if oee >= 0.68:
        return "mid"
    return "low"

# ── Operator master ─────────────────────────────────────────────────────────
N_OPERATORS      = 20
N_MACHINISTS     = 16
OPERATOR_IDS     = [f"OPR-{i:03d}" for i in range(1, N_OPERATORS + 1)]
MACHINIST_IDS    = OPERATOR_IDS[:N_MACHINISTS]
MAINT_TECH_IDS   = OPERATOR_IDS[N_MACHINISTS:]

# The HR system of record carries both the current OPR identifier and the
# legacy payroll number. Machine and job records reference the payroll number
# only, so the two identifier schemes are reconciled through this master.
EMPLOYEE_NUMBER_BY_OPERATOR = {
    "OPR-001": 1042, "OPR-002": 1067, "OPR-003": 1013, "OPR-004": 1088,
    "OPR-005": 1025, "OPR-006": 1071, "OPR-007": 1009, "OPR-008": 1054,
    "OPR-009": 1096, "OPR-010": 1031, "OPR-011": 1078, "OPR-012": 1005,
    "OPR-013": 1063, "OPR-014": 1019, "OPR-015": 1085, "OPR-016": 1048,
    "OPR-017": 1092, "OPR-018": 1037, "OPR-019": 1074, "OPR-020": 1051,
}

CERTIFICATIONS = ["CNC Level I", "CNC Level II", "Setup"]

# Certification handling for two specific operators, keyed to the scoring window.
CERT_EXPIRING_OPERATOR = "OPR-004"   # lapses within 30 days of the scoring window
CERT_LAPSED_OPERATOR   = "OPR-009"   # already lapsed

# ── Preventive-maintenance schedule ─────────────────────────────────────────
PM_INTERVAL_DAYS         = 42
PM_ONTIME_RATE           = 0.78
PM_ONTIME_RATE_AGING     = 0.58
PM_ADHOC_RATE            = 0.06   # completions logged without a scheduled date
PM_OVERDUE_THRESHOLD_DAYS = 14
# Every machine drifts past its PM interval at least occasionally. This floor
# guarantees a minimum number of materially overdue (>14 day) PM periods per
# machine so the PM-overdue vs alarm-rate relationship is observable within each
# machine, not just across the fleet.
MIN_OVERDUE_PMS_PER_MACHINE = 3

# ── Maintenance event coding ────────────────────────────────────────────────
MAINTENANCE_TYPES = ["PLANNED_PM", "UNPLANNED_REPAIR", "INSPECTION"]
FAILURE_CODES         = ["TOOLING", "MECHANICAL", "ELECTRICAL",
                         "OPERATOR_INDUCED", "ENVIRONMENTAL"]
FAILURE_CODE_WEIGHTS  = [0.40, 0.30, 0.15, 0.10, 0.05]

# ── Alarm coding ────────────────────────────────────────────────────────────
ALARM_CODES = [
    "ALM-1010 Spindle Overload",
    "ALM-1042 Axis Servo Fault",
    "ALM-2003 Coolant Pressure Low",
    "ALM-2110 Tool Life Expired",
    "ALM-3001 Overtravel Limit",
    "ALM-3220 Lubrication Low",
    "ALM-4005 Hydraulic Pressure Fault",
]

# ── Work-order attributes ───────────────────────────────────────────────────
CUSTOMERS        = [f"CUST-{i:02d}" for i in range(1, 9)]
CUSTOMER_WEIGHTS = [0.28, 0.22, 0.16, 0.12, 0.09, 0.07, 0.04, 0.02]
MATERIAL_TYPES   = ["Aluminum", "Steel", "Stainless", "Titanium"]
MATERIAL_WEIGHTS = [0.40, 0.35, 0.20, 0.05]
JOB_STATUSES     = ["COMPLETE", "IN_PROGRESS", "ON_HOLD"]

WORK_ORDERS_PER_DAY_MIN = 6
WORK_ORDERS_PER_DAY_MAX = 16
# Setup time booked per job in the ERP. Calibrated against the machine-state
# log: the fleet books roughly 40 minutes of SETUP state per machine per day
# across about 1.5 jobs, so a job carries about 25 minutes of setup. Setup
# discipline varies machinist to machinist; a few run materially longer.
SETUP_HOURS_MEDIAN           = 0.42   # about 25 minutes
SETUP_HOURS_JOB_STD          = 0.08   # job-to-job variation around an operator's own level
SETUP_OPERATOR_SPREAD_STD    = 0.08   # operator-to-operator variation in setup discipline
SETUP_OPERATOR_SPREAD_CLIP   = (0.85, 1.15)

# ── Reliability relationships ───────────────────────────────────────────────
# Ratios the generated data is calibrated to express once source systems are
# joined. Named constants so the calibration lives in one place.
AGING_ASSETS                        = ["MCH-007", "MCH-011"]
AGING_ASSET_FAILURE_MULTIPLIER      = 2.8
# Both shifts see elevated unplanned stoppages during their first N minutes
# (cold starts, warm-ups, first-piece setups). The elevation is a shift-start
# phenomenon common to both crews, with Shift B only modestly worse because it
# also picks up a production run already in progress.
SHIFT_STARTUP_WINDOW_MINUTES        = 45
SHIFT_STARTUP_DOWNTIME_MULTIPLIER   = {"A": 1.2, "B": 1.3}
# Alarm rate on the two aging assets, expressed as a multiple of the mid-tier
# peer alarm decision weight rather than of the fleet mean. Aging equipment runs
# noisier than its healthier peers, but only modestly: this lands the realised
# alarm rate per running hour at roughly 1.3x to 1.5x the mid-tier machines
# (MCH-005/006/008/009/010) rather than the several-times gap a raw failure
# multiplier would produce.
AGING_ASSET_ALARM_MULTIPLIER        = 1.35
# Alarm decision-weight multiplier while a machine is materially past a due PM.
# Set slightly below the 2.4x target because raising alarm probability also
# reduces running time, so the realised alarm rate per running hour lands near
# 2.4x rather than at the raw multiplier.
PM_OVERDUE_ALARM_MULTIPLIER         = 2.25
EXTENDED_SETUP_OPERATORS            = ["OPR-004", "OPR-009", "OPR-012"]
# These three sit above the cohort on setup time, but within the range a
# coaching conversation closes rather than as extreme outliers. Expressed
# against the base median; the reported ratio compares against the cohort
# median of all booked jobs, which their own jobs lift slightly, so these
# factors land the reported ratio at roughly 1.25x to 1.4x.
EXTENDED_SETUP_RATIO_RANGE          = (1.34, 1.48)


def operating_days():
    """Return every Monday–Saturday date within the observation window."""
    days, d = [], START_DATE
    while d <= END_DATE:
        if d.weekday() < 6:   # Monday=0 … Saturday=5
            days.append(d)
        d += timedelta(days=1)
    return days
