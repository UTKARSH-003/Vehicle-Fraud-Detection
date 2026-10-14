from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = APP_ROOT / "data" / "fraud_oracle.csv"
MODEL_DIR = APP_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_bundle.joblib"
MODEL_VERSION = "1.3"
TARGET_COLUMN = "FraudFound_P"

# Identifiers, investigator assignment, and demographic attributes are not used
# to make a triage score. They are not necessary for claim-level evidence review.
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "PolicyNumber",
    "RepNumber",
    "Sex",
    "Age",
    "MaritalStatus",
    "AgeOfPolicyHolder",
}

FEATURE_LABELS = {
    "Month": "Accident month",
    "WeekOfMonth": "Accident week of month",
    "DayOfWeek": "Accident day",
    "Make": "Vehicle make",
    "AccidentArea": "Accident area",
    "DayOfWeekClaimed": "Claim day",
    "MonthClaimed": "Claim month",
    "WeekOfMonthClaimed": "Claim week of month",
    "Fault": "Fault assigned to",
    "PolicyType": "Policy type",
    "VehicleCategory": "Vehicle category",
    "VehiclePrice": "Vehicle price band",
    "Deductible": "Deductible",
    "DriverRating": "Driver rating",
    "Days_Policy_Accident": "Policy-to-accident interval",
    "Days_Policy_Claim": "Policy-to-claim interval",
    "PastNumberOfClaims": "Past claims",
    "AgeOfVehicle": "Vehicle age",
    "PoliceReportFiled": "Police report filed",
    "WitnessPresent": "Witness present",
    "AgentType": "Agent type",
    "NumberOfSuppliments": "Claim supplements",
    "AddressChange_Claim": "Time since address change",
    "NumberOfCars": "Cars on policy",
    "Year": "Accident year",
    "BasePolicy": "Base policy",
}

FEATURE_GROUPS = {
    "Accident & timing": {
        "Month",
        "WeekOfMonth",
        "DayOfWeek",
        "AccidentArea",
        "DayOfWeekClaimed",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "Year",
        "Fault",
    },
    "Vehicle & cover": {
        "Make",
        "PolicyType",
        "VehicleCategory",
        "VehiclePrice",
        "AgeOfVehicle",
        "BasePolicy",
        "Deductible",
        "NumberOfCars",
    },
    "Claim history": {
        "Days_Policy_Accident",
        "Days_Policy_Claim",
        "PastNumberOfClaims",
        "AddressChange_Claim",
        "NumberOfSuppliments",
    },
    "Evidence & handling": {
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "DriverRating",
    },
}

CATEGORY_ORDER = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    "MonthClaimed": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    "DayOfWeek": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ],
    "DayOfWeekClaimed": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ],
}

CAVEATS = [
    "This is a historical benchmark dataset, not a current insurer's claims population; validate locally before operational use.",
    "The source data has no claim-amount or policy-tenure field, so neither is modeled or scored.",
    "Policy, representative, and demographic identifiers are excluded from model inputs.",
    "The model score is not calibrated as a real-world probability.",
    "A score is a review aid only. It is not proof of fraud and must not be used alone to deny or delay a claim.",
]
