"""Clinical reference data: vital-sign metadata and physiological plausibility limits.

Keeping these in one place means the UI, the simulator, the feature builder, and
the alert rules all agree on what "normal" means and what units they are in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VitalSpec:
    """Everything the system needs to know about one vital-sign channel."""

    key: str
    display_name: str
    short_name: str
    unit: str
    normal_low: float
    normal_high: float
    # Hard physiological bounds. Values outside are treated as artefact and dropped.
    plausible_low: float
    plausible_high: float
    # Axis range used by trend charts, so a flat trace does not fill the panel.
    axis_low: float
    axis_high: float
    decimals: int = 0
    # Categorical palette slot (1-indexed, fixed order - never cycled).
    series_slot: int = 1

    def is_plausible(self, value: float | None) -> bool:
        if value is None:
            return False
        return self.plausible_low <= value <= self.plausible_high

    def is_normal(self, value: float | None) -> bool:
        if value is None:
            return False
        return self.normal_low <= value <= self.normal_high

    def format(self, value: float | None) -> str:
        if value is None:
            return "--"
        return f"{value:.{self.decimals}f}"

    def format_with_unit(self, value: float | None) -> str:
        if value is None:
            return "--"
        return f"{self.format(value)} {self.unit}".strip()


VITAL_SPECS: dict[str, VitalSpec] = {
    "heart_rate": VitalSpec(
        key="heart_rate",
        display_name="Heart rate",
        short_name="HR",
        unit="bpm",
        normal_low=51,
        normal_high=90,
        plausible_low=20,
        plausible_high=250,
        axis_low=40,
        axis_high=160,
        series_slot=1,
    ),
    "spo2": VitalSpec(
        key="spo2",
        display_name="Oxygen saturation",
        short_name="SpO₂",
        unit="%",
        normal_low=96,
        normal_high=100,
        plausible_low=50,
        plausible_high=100,
        axis_low=80,
        axis_high=100,
        series_slot=3,
    ),
    "bp_systolic": VitalSpec(
        key="bp_systolic",
        display_name="Systolic pressure",
        short_name="SBP",
        unit="mmHg",
        normal_low=111,
        normal_high=219,
        plausible_low=40,
        plausible_high=300,
        axis_low=70,
        axis_high=180,
        series_slot=5,
    ),
    "bp_diastolic": VitalSpec(
        key="bp_diastolic",
        display_name="Diastolic pressure",
        short_name="DBP",
        unit="mmHg",
        normal_low=60,
        normal_high=90,
        plausible_low=20,
        plausible_high=200,
        axis_low=40,
        axis_high=110,
        series_slot=5,
    ),
    "resp_rate": VitalSpec(
        key="resp_rate",
        display_name="Respiratory rate",
        short_name="RR",
        unit="/min",
        normal_low=12,
        normal_high=20,
        plausible_low=4,
        plausible_high=60,
        axis_low=6,
        axis_high=40,
        series_slot=2,
    ),
    "temperature": VitalSpec(
        key="temperature",
        display_name="Temperature",
        short_name="Temp",
        unit="°C",
        normal_low=36.1,
        normal_high=38.0,
        plausible_low=28.0,
        plausible_high=43.0,
        axis_low=34.0,
        axis_high=41.0,
        decimals=1,
        series_slot=4,
    ),
}

#: Channels the bedside model consumes, in a fixed order.
CORE_VITALS: tuple[str, ...] = (
    "heart_rate",
    "spo2",
    "bp_systolic",
    "resp_rate",
    "temperature",
)

#: Channels rendered on the multi-signal trend chart, in draw order.
TREND_VITALS: tuple[str, ...] = (
    "heart_rate",
    "spo2",
    "bp_systolic",
    "resp_rate",
    "temperature",
)

#: PhysioNet Challenge 2012 parameter names mapped onto our channel keys.
PHYSIONET_PARAMETER_MAP: dict[str, str] = {
    "HR": "heart_rate",
    "SaO2": "spo2",
    "SysABP": "bp_systolic",
    "NISysABP": "bp_systolic",
    "DiasABP": "bp_diastolic",
    "NIDiasABP": "bp_diastolic",
    "RespRate": "resp_rate",
    "Temp": "temperature",
    "GCS": "gcs",
    "MAP": "map",
    "NIMAP": "map",
    "FiO2": "fio2",
    "MechVent": "mech_vent",
    "Urine": "urine",
}

#: Static, per-stay descriptors recorded once at admission.
PHYSIONET_STATIC_PARAMETERS: tuple[str, ...] = (
    "Age",
    "Gender",
    "Height",
    "Weight",
    "ICUType",
)

ICU_TYPE_LABELS: dict[int, str] = {
    1: "Coronary Care Unit",
    2: "Cardiac Surgery Recovery",
    3: "Medical ICU",
    4: "Surgical ICU",
}

#: Diagnoses used for the synthetic ward. Plausible ICU admission reasons.
DEMO_DIAGNOSES: tuple[str, ...] = (
    "Community-acquired pneumonia",
    "Post-operative CABG recovery",
    "Septic shock, urinary source",
    "Acute decompensated heart failure",
    "COPD exacerbation, type 2 failure",
    "Diabetic ketoacidosis",
    "Acute pancreatitis",
    "Traumatic brain injury, observation",
    "Upper GI haemorrhage",
    "Aspiration pneumonitis",
    "Acute kidney injury on sepsis",
    "Status post cardiac arrest",
)

#: Pseudonymous names for the demo ward. No real patient data is used anywhere.
DEMO_NAMES: tuple[str, ...] = (
    "Asha Rao",
    "Ben Oyelaran",
    "Carla Mendes",
    "Dinesh Kumar",
    "Elena Petrova",
    "Farid Haddad",
    "Grace Kimani",
    "Hiroshi Tanaka",
    "Ines Duarte",
    "Jonas Weber",
    "Kavya Nair",
    "Liam Doherty",
    "Mei Chen",
    "Nadia Rahman",
    "Omar Salah",
    "Priya Iyer",
    "Quentin Blair",
    "Rosa Alvarez",
    "Sven Larsen",
    "Tara Nolan",
    "Umar Sesay",
    "Vera Kowalski",
    "Wei Zhang",
    "Yusuf Demir",
)

__all__ = [
    "CORE_VITALS",
    "DEMO_DIAGNOSES",
    "DEMO_NAMES",
    "ICU_TYPE_LABELS",
    "PHYSIONET_PARAMETER_MAP",
    "PHYSIONET_STATIC_PARAMETERS",
    "TREND_VITALS",
    "VITAL_SPECS",
    "VitalSpec",
]
