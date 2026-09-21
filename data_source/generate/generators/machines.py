"""Machine catalog from the MachineMetrics equipment register."""
import pandas as pd

from ..config import MACHINES_DATA, FLEET_REFERENCE_DATE


def generate_machines() -> pd.DataFrame:
    records = []
    for machine_id, machine_type, controller, cell, age_years, max_rpm in MACHINES_DATA:
        install_date = FLEET_REFERENCE_DATE.replace(year=FLEET_REFERENCE_DATE.year - age_years)
        records.append({
            "machine_id":       machine_id,
            "machine_type":     machine_type,
            "controller_type":  controller,
            "location_cell":    cell,
            "machine_age_years": age_years,
            "max_spindle_rpm":  max_rpm,
            "install_date":     install_date.isoformat(),
        })
    return pd.DataFrame(records)
