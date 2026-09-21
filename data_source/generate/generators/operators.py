"""Operator records from the ADP HR export."""
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

from ..config import (
    RANDOM_SEED, OPERATOR_IDS, MACHINIST_IDS, MAINT_TECH_IDS,
    EMPLOYEE_NUMBER_BY_OPERATOR, CERTIFICATIONS,
    CERT_EXPIRING_OPERATOR, CERT_LAPSED_OPERATOR, SCORING_WINDOW_START,
)

HIRE_START = date(2015, 1, 1)
HIRE_END   = date(2023, 12, 31)


def _shift_for(operator_id: str) -> str:
    """Balance each role evenly across the two shifts."""
    if operator_id in MACHINIST_IDS:
        idx, size = MACHINIST_IDS.index(operator_id), len(MACHINIST_IDS)
    else:
        idx, size = MAINT_TECH_IDS.index(operator_id), len(MAINT_TECH_IDS)
    return "Shift A" if idx < size // 2 else "Shift B"


def _expiry_for(operator_id: str, rng: random.Random) -> date:
    if operator_id == CERT_EXPIRING_OPERATOR:
        return SCORING_WINDOW_START + timedelta(days=rng.randint(5, 28))
    if operator_id == CERT_LAPSED_OPERATOR:
        return date(2025, 8, 15)
    return SCORING_WINDOW_START + timedelta(days=rng.randint(150, 700))


def generate_operators() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    faker = Faker()
    Faker.seed(RANDOM_SEED)

    hire_span = (HIRE_END - HIRE_START).days
    records = []
    for operator_id in OPERATOR_IDS:
        role = "Machinist" if operator_id in MACHINIST_IDS else "Maintenance Tech"
        hire_date = HIRE_START + timedelta(days=rng.randint(0, hire_span))
        records.append({
            "operator_id":               operator_id,
            "operator_name":             faker.name(),
            "employee_number":           EMPLOYEE_NUMBER_BY_OPERATOR[operator_id],
            "shift":                     _shift_for(operator_id),
            "role":                      role,
            "hire_date":                 hire_date.isoformat(),
            "certification":             rng.choice(CERTIFICATIONS),
            "certification_expiry_date": _expiry_for(operator_id, rng).isoformat(),
        })
    return pd.DataFrame(records)
