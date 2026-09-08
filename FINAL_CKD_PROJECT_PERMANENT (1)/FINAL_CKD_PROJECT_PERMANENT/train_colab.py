# ============================================================
# FINAL COLAB CODE
# PRIVACY-PRESERVING FEDERATED LEARNING FRAMEWORK
# FOR MULTI-HOSPITAL MEDICAL DATA ANALYSIS
# CKD CASE STUDY
# ============================================================

# Colab install: !pip -q install ucimlrepo kagglehub torch scikit-learn pandas numpy matplotlib reportlab joblib

import os, re, json, random, warnings, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from ucimlrepo import fetch_ucirepo
import kagglehub

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TITLE = (
    "PRIVACY-PRESERVING FEDERATED LEARNING FRAMEWORK "
    "FOR MULTI-HOSPITAL MEDICAL DATA ANALYSIS"
)

OUT = Path("/content/CKD_Federated_Project_Output")

for folder in [
    OUT,
    OUT / "model",
    OUT / "results",
    OUT / "graphs",
    OUT / "reports"
]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Output:", OUT)

# ============================================================
# 2. FEATURES
# ============================================================

FEATURES = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc",
    "appet", "pe", "ane", "htn", "dm", "cad"
]

NUMERIC = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]

CATEGORICAL = [
    "rbc", "pc", "pcc", "ba",
    "appet", "pe", "ane",
    "htn", "dm", "cad"
]

RANGES = {
    "age": (0, 100),
    "bp": (40, 220),
    "sg": (1.005, 1.030),
    "al": (0, 5),
    "su": (0, 5),
    "bgr": (40, 500),
    "bu": (0, 300),
    "sc": (0, 20),
    "sod": (100, 180),
    "pot": (2, 10),
    "hemo": (3, 20),
    "pcv": (10, 60),
    "wbcc": (1000, 30000),
    "rbcc": (1, 10)
}

CAT_MAP = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1}
}

ALIASES = {
    "age": ["age"],
    "bp": ["bp", "bloodpressure"],
    "sg": ["sg", "urinespecificgravity", "specificgravity"],
    "al": ["al", "albumin", "urinealbumin"],
    "su": ["su", "sugar", "urinesugar"],
    "rbc": ["rbc", "redbloodcells", "redbloodcell"],
    "pc": ["pc", "puscells", "puscell"],
    "pcc": ["pcc", "puscellclumps", "puscellclump"],
    "ba": ["ba", "bacteria"],
    "bgr": [
        "bgr", "randombloodglucose",
        "bloodglucose", "glucose"
    ],
    "bu": ["bu", "bloodurea", "urea"],
    "sc": ["sc", "serumcreatinine", "creatinine"],
    "sod": ["sod", "bloodsodium", "sodium"],
    "pot": ["pot", "bloodpotassium", "potassium"],
    "hemo": ["hemo", "hemoglobin"],
    "pcv": ["pcv", "packedcellvolume"],
    "wbcc": [
        "wbcc", "whitebloodcellcount",
        "wbc", "whitebloodcells"
    ],
    "rbcc": [
        "rbcc", "redbloodcellcount",
        "rbc_count", "redbloodcells"
    ],
    "appet": ["appet", "appetite"],
    "pe": ["pe", "pedema", "edema"],
    "ane": ["ane", "anemia"],
    "htn": ["htn", "hypertension"],
    "dm": [
        "dm", "diabetesmellitus",
        "diabetes"
    ],
    "cad": [
        "cad", "coronaryarterydisease"
    ]
}

# ============================================================
# 3. TARGET MAPPING
# ============================================================

def normalize_column_name(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def map_target(series):
    result = []

    for value in series:

        raw = str(value).strip().lower()
        compact = re.sub(r"[^a-z0-9]", "", raw)

        positive = {
            "ckd",
            "yes",
            "y",
            "1",
            "true",
            "positive",
            "affected",
            "disease",
            "present",
            "chronickidneydisease"
        }

        negative = {
            "notckd",
            "nockd",
            "no",
            "n",
            "0",
            "false",
            "negative",
            "notaffected",
            "normal",
            "healthy",
            "nockddisease"
        }

        if compact in positive:
            result.append(1)

        elif compact in negative:
            result.append(0)

        else:
            try:
                number = float(raw)

                if number > 0:
                    result.append(1)
                else:
                    result.append(0)

            except:
                result.append(np.nan)

    return pd.Series(
        result,
        index=series.index,
        dtype="float"
    )


def find_column(df, key):

    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for alias in ALIASES[key]:

        alias_norm = normalize_column_name(alias)

        if alias_norm in normalized:
            return normalized[alias_norm]

    return None


def harmonize_dataset(X, y, source_name):

    X = X.copy()

    X.columns = [
        str(c).strip().lower()
        for c in X.columns
    ]

    X = X.reset_index(drop=True)

    y = pd.Series(y).reset_index(drop=True)

    # Keep an already-binary numeric target unchanged; otherwise map text labels.
    numeric_y = pd.to_numeric(y, errors="coerce")
    non_null_numeric = numeric_y.dropna()
    if len(non_null_numeric) > 0 and set(non_null_numeric.unique()).issubset({0, 1}):
        y = numeric_y.astype(float)
    else:
        y = map_target(y)

    output = pd.DataFrame(
        index=X.index
    )

    for feature in FEATURES:

        column = find_column(X, feature)

        if column is not None:
            output[feature] = X[column]
        else:
            output[feature] = np.nan

    output["target"] = y.values

    output = output.dropna(
        subset=["target"]
    ).reset_index(drop=True)

    if output["target"].nunique() != 2:

        raise RuntimeError(
            source_name +
            ": target does not contain both classes."
        )

    output["target"] = output["target"].astype(int)

    print(
        source_name,
        "->",
        output.shape,
        "CKD=",
        int(output.target.sum()),
        "Not CKD=",
        int((output.target == 0).sum())
    )

    return output

# ============================================================
# 4. LOAD UCI CKD DATASET
# ============================================================

print("\nLoading UCI CKD Dataset...")

uci1 = fetch_ucirepo(id=336)

D1 = harmonize_dataset(
    uci1.data.features,
    uci1.data.targets.iloc[:, 0],
    "UCI CKD Dataset 336"
)

# ============================================================
# 5. LOAD KAGGLE CKD DATASET
# ============================================================

print("\nDownloading Kaggle CKD Dataset...")

kaggle_path = kagglehub.dataset_download(
    "marcklugerhart/chronic-kidney-failure-dataset-reboot"
)

csv_files = list(
    Path(kaggle_path).rglob("*.csv")
)

if len(csv_files) == 0:

    raise FileNotFoundError(
        "No CSV file found in Kaggle dataset."
    )

print("Kaggle CSV:", csv_files[0])

KAGGLE_RAW = pd.read_csv(csv_files[0])

print(
    "Kaggle shape:",
    KAGGLE_RAW.shape
)

# ------------------------------------------------------------
# Detect target column robustly
# ------------------------------------------------------------

target_column = None

# Strongest known column name for the Kaggle Reboot dataset.
preferred_names = {
    "chronickidneydisease",
    "ckd",
    "classification",
    "target",
    "class",
    "label",
    "outcome"
}

# First pass: semantic name + valid two-class mapping.
for column in KAGGLE_RAW.columns:
    normalized = normalize_column_name(column)
    mapped = map_target(KAGGLE_RAW[column])
    if normalized in preferred_names or "chronickidneydisease" in normalized:
        if mapped.notna().sum() > 0 and mapped.nunique() == 2:
            target_column = column
            break

# Second pass: scan every column for CKD-like binary values.
if target_column is None:
    candidates = []
    for column in KAGGLE_RAW.columns:
        mapped = map_target(KAGGLE_RAW[column])
        valid = mapped.dropna()
        if len(valid) == 0 or valid.nunique() != 2:
            continue
        name_score = 0
        n = normalize_column_name(column)
        if "kidney" in n or "ckd" in n or "class" in n or "target" in n or "disease" in n:
            name_score += 10
        candidates.append((name_score, column, int(len(valid))))
    if candidates:
        candidates.sort(reverse=True)
        target_column = candidates[0][1]

if target_column is None:
    print("Kaggle columns:", list(KAGGLE_RAW.columns))
    raise RuntimeError("Could not identify a valid binary CKD target column in Kaggle CKD Reboot dataset.")

print("Kaggle target column:", target_column)
print("Kaggle raw target values:", KAGGLE_RAW[target_column].dropna().astype(str).str.strip().value_counts().to_dict())
print("Kaggle mapped target values:", map_target(KAGGLE_RAW[target_column]).value_counts(dropna=False).to_dict())

D2 = harmonize_dataset(
    KAGGLE_RAW.drop(
        columns=[target_column]
    ),
    KAGGLE_RAW[target_column],
    "Kaggle CKD Reboot"
)

# ============================================================
# 6. LOAD UCI RISK FACTOR CKD DATASET
# ============================================================

print("\nLoading UCI Risk Factor CKD Dataset...")

uci3 = fetch_ucirepo(id=857)

D3 = harmonize_dataset(
    uci3.data.features,
    uci3.data.targets.iloc[:, 0],
    "UCI Risk Factor CKD 857"
)

# ============================================================
# 7. CREATE FOUR SIMULATED HOSPITALS
# ============================================================

KAGGLE_H1, KAGGLE_H2 = train_test_split(
    D2,
    test_size=0.5,
    stratify=D2["target"],
    random_state=SEED
)

KAGGLE_H1 = KAGGLE_H1.reset_index(drop=True)
KAGGLE_H2 = KAGGLE_H2.reset_index(drop=True)

HOSPITAL_RAW = {

    "Hospital 1": D1.reset_index(drop=True),

    "Hospital 2": KAGGLE_H1,

    "Hospital 3": KAGGLE_H2,

    "Hospital 4": D3.reset_index(drop=True)
}

# ============================================================
# 8. MISSING FEATURE CONFIGURATION
# ============================================================

HOSPITAL_CONFIG = {

    "Hospital 1": {
        "missing": 0.05
    },

    "Hospital 2": {
        "missing": 0.15
    },

    "Hospital 3": {
        "missing": 0.25
    },

    "Hospital 4": {
        "missing": 0.40
    }
}

# ============================================================
# 9. ADAPTIVE PRIVACY MECHANISM
# ============================================================

def adaptive_noise(missing_level):

    if missing_level <= 0.05:
        return 0.05

    elif missing_level <= 0.15:
        return 0.10

    elif missing_level <= 0.25:
        return 0.20

    else:
        return 0.30


NOISE_TABLE = pd.DataFrame([
    {
        "Hospital": hospital,
        "Missing Level":
            HOSPITAL_CONFIG[hospital]["missing"],
        "Adaptive Noise":
            adaptive_noise(
                HOSPITAL_CONFIG[hospital]["missing"]
            )
    }

    for hospital in HOSPITAL_CONFIG
])

print("\nAdaptive Privacy Mapping")
print(NOISE_TABLE)

# ============================================================
# 10. INJECT MISSING FEATURES
# ============================================================

def inject_missing(
    dataframe,
    missing_rate,
    seed
):

    rng = np.random.default_rng(seed)

    result = dataframe.copy()

    total_cells = (
        len(result) *
        len(FEATURES)
    )

    number_missing = int(
        total_cells *
        missing_rate
    )

    if number_missing > 0:

        selected = rng.choice(
            total_cells,
            number_missing,
            replace=False
        )

        for position in selected:

            row_index, feature_index = divmod(
                int(position),
                len(FEATURES)
            )

            feature = FEATURES[
                feature_index
            ]

            result.loc[
                row_index,
                feature
            ] = np.nan

    return result

# ============================================================
# 11. GLOBAL MEDIANS
# ============================================================

all_features = pd.concat(
    [
        dataframe[FEATURES]
        for dataframe in HOSPITAL_RAW.values()
    ],
    ignore_index=True
)

MEDIANS = {}

for feature in NUMERIC:

    MEDIANS[feature] = pd.to_numeric(
        all_features[feature],
        errors="coerce"
    ).median()

# ============================================================
# 12. ENCODING
# ============================================================

def encode(dataframe):

    result = pd.DataFrame(
        index=dataframe.index
    )

    # Numeric features
    for feature in NUMERIC:

        result[feature] = pd.to_numeric(
            dataframe[feature],
            errors="coerce"
        ).fillna(
            MEDIANS[feature]
        ).astype(float)

    # Categorical features
    for feature in CATEGORICAL:

        values = (
            dataframe[feature]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        result[feature] = (
            values
            .map(CAT_MAP[feature])
            .fillna(0)
            .astype(float)
        )

    # Normalize numerical values
    for feature in NUMERIC:

        low, high = RANGES[feature]

        result[feature] = (
            result[feature] - low
        ) / (high - low)

    result = result.replace(
        [np.inf, -np.inf],
        0
    )

    result = result.fillna(0)

    return result[FEATURES]

# ============================================================
# 13. CREATE TRAIN / TEST DATA
# ============================================================

TRAIN_DATA = {}
TEST_DATA = {}

for index, (
    hospital,
    dataframe
) in enumerate(HOSPITAL_RAW.items()):

    missing_rate = (
        HOSPITAL_CONFIG[
            hospital
        ]["missing"]
    )

    dataframe_missing = inject_missing(
        dataframe,
        missing_rate,
        SEED + index
    )

    train_df, test_df = train_test_split(
        dataframe_missing,
        test_size=0.20,
        stratify=dataframe_missing["target"],
        random_state=SEED
    )

    train_x = encode(
        train_df[FEATURES]
    ).values.astype(np.float32)

    train_y = train_df[
        "target"
    ].values.astype(np.float32)

    test_x = encode(
        test_df[FEATURES]
    ).values.astype(np.float32)

    test_y = test_df[
        "target"
    ].values.astype(np.float32)

    TRAIN_DATA[hospital] = (
        train_x,
        train_y
    )

    TEST_DATA[hospital] = (
        test_x,
        test_y
    )

    print(
        hospital,
        "Train:",
        len(train_y),
        "Test:",
        len(test_y),
        "Missing:",
        missing_rate
    )

# ============================================================
# 14. COMBINE HOSPITAL DATA
# ============================================================

def combined_data(split="test"):

    dataset = (
        TRAIN_DATA
        if split == "train"
        else TEST_DATA
    )

    X = np.concatenate(
        [
            dataset[h][0]
            for h in dataset
        ]
    )

    y = np.concatenate(
        [
            dataset[h][1]
            for h in dataset
        ]
    )

    return X, y

# ============================================================
# 15. MODEL
# ============================================================

class CKDModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(24, 64),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(1)

# ============================================================
# 16. MODEL STATE FUNCTIONS
# ============================================================

def get_state(model):

    return {
        key:
        value.detach()
        .cpu()
        .clone()

        for key, value
        in model.state_dict().items()
    }


def set_state(model, model_state):

    model.load_state_dict(
        model_state
    )

    return model

# ============================================================
# 17. FEDERATED LEARNING PARAMETERS
# ============================================================

ROUNDS = 8

LOCAL_EPOCHS = 4

LEARNING_RATE = 0.003

BATCH_SIZE = 32

CLIP_NORM = 1.0

FIXED_NOISE = 0.15

# ============================================================
# 18. LOCAL TRAINING + PRIVACY PROTECTION
# ============================================================

def local_update(
    global_state,
    X,
    y,
    noise
):

    model = CKDModel().to(
        DEVICE
    )

    set_state(
        model,
        global_state
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model.train()

    for epoch in range(
        LOCAL_EPOCHS
    ):

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            output = model(xb)

            loss = loss_function(
                output,
                yb
            )

            loss.backward()

            optimizer.step()

    trained_state = get_state(
        model
    )

    protected_state = {}

    for key in global_state:

        delta = (
            trained_state[key]
            -
            global_state[key]
        )

        # Gradient/update clipping
        norm = torch.norm(
            delta
        )

        if norm > CLIP_NORM:

            delta = (
                delta *
                (
                    CLIP_NORM /
                    (norm + 1e-12)
                )
            )

        # Gaussian privacy noise
        if noise > 0:

            delta = (
                delta +
                torch.randn_like(delta)
                * noise
            )

        protected_state[key] = (
            global_state[key]
            +
            delta
        )

    return protected_state

# ============================================================
# 19. FEDAVG
# ============================================================

def fedavg(
    client_states,
    client_weights
):

    total_weight = sum(
        client_weights
    )

    global_state = {}

    for key in client_states[0]:

        weighted_sum = sum(
            state[key] * weight
            for state, weight
            in zip(
                client_states,
                client_weights
            )
        )

        global_state[key] = (
            weighted_sum /
            total_weight
        )

    return global_state

# ============================================================
# 20. EVALUATION
# ============================================================

def evaluate(
    model_state,
    X,
    y
):

    model = CKDModel().to(
        DEVICE
    )

    set_state(
        model,
        model_state
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            torch.tensor(
                X,
                dtype=torch.float32,
                device=DEVICE
            )
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "Accuracy":
            accuracy_score(
                y,
                predictions
            ),

        "Precision":
            precision_score(
                y,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y,
                predictions,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y,
                probabilities
            )
            if len(np.unique(y)) == 2
            else 0.5
    }

    return (
        metrics,
        probabilities,
        predictions
    )

# ============================================================
# 21. RUN FEDERATED LEARNING
# ============================================================

def run_federated_learning(
    method
):

    initial_model = CKDModel()

    global_state = get_state(
        initial_model
    )

    history = []

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        client_states = []

        client_weights = []

        for hospital in TRAIN_DATA:

            if method == "normal":

                noise = 0.0

            elif method == "fixed":

                noise = FIXED_NOISE

            else:

                missing = (
                    HOSPITAL_CONFIG[
                        hospital
                    ]["missing"]
                )

                noise = adaptive_noise(
                    missing
                )

            X, y = TRAIN_DATA[
                hospital
            ]

            updated_state = local_update(
                global_state,
                X,
                y,
                noise
            )

            client_states.append(
                updated_state
            )

            client_weights.append(
                len(y)
            )

        global_state = fedavg(
            client_states,
            client_weights
        )

        X_test, y_test = combined_data(
            "test"
        )

        metrics, _, _ = evaluate(
            global_state,
            X_test,
            y_test
        )

        history.append(
            {
                "Round":
                    round_number,
                **metrics
            }
        )

        print(
            method,
            "Round",
            round_number,
            "| Accuracy:",
            round(
                metrics["Accuracy"],
                4
            ),
            "| F1:",
            round(
                metrics["F1"],
                4
            )
        )

    return (
        global_state,
        pd.DataFrame(history)
    )

# ============================================================
# 22. THREE FL EXPERIMENTS
# ============================================================

print("\n==============================")
print("NORMAL FEDERATED LEARNING")
print("==============================")

NORMAL_STATE, NORMAL_HISTORY = (
    run_federated_learning(
        "normal"
    )
)

print("\n==============================")
print("FIXED DP FEDERATED LEARNING")
print("==============================")

FIXED_STATE, FIXED_HISTORY = (
    run_federated_learning(
        "fixed"
    )
)

print("\n==============================")
print("PROPOSED ADAPTIVE DP FL")
print("==============================")

ADAPTIVE_STATE, ADAPTIVE_HISTORY = (
    run_federated_learning(
        "adaptive"
    )
)

# ============================================================
# 23. FINAL PERFORMANCE
# ============================================================

def final_row(
    state,
    method_name
):

    metrics, _, _ = evaluate(
        state,
        *combined_data("test")
    )

    return {
        "Method":
            method_name,
        **metrics
    }


FINAL_RESULTS = pd.DataFrame([

    final_row(
        NORMAL_STATE,
        "Normal FL"
    ),

    final_row(
        FIXED_STATE,
        "Fixed-DP FL"
    ),

    final_row(
        ADAPTIVE_STATE,
        "Proposed Adaptive-DP FL"
    )

])

print("\n==============================")
print("FINAL PERFORMANCE")
print("==============================")

display(
    FINAL_RESULTS.round(4)
)

# ============================================================
# 24. HOSPITAL-WISE RESULTS
# ============================================================

hospital_results = []

for method_name, state in [

    ("Normal FL", NORMAL_STATE),

    ("Fixed-DP FL", FIXED_STATE),

    (
        "Proposed Adaptive-DP FL",
        ADAPTIVE_STATE
    )
]:

    for hospital in TEST_DATA:

        metrics, _, _ = evaluate(
            state,
            *TEST_DATA[hospital]
        )

        hospital_results.append(
            {
                "Method":
                    method_name,

                "Hospital":
                    hospital,

                **metrics
            }
        )

HOSPITAL_RESULTS = pd.DataFrame(
    hospital_results
)

print("\n==============================")
print("HOSPITAL-WISE RESULTS")
print("==============================")

display(
    HOSPITAL_RESULTS.round(4)
)

# ============================================================
# 25. MEMBERSHIP INFERENCE ATTACK
# ============================================================

def confidence_scores(
    model_state,
    X
):

    model = CKDModel().to(
        DEVICE
    )

    set_state(
        model,
        model_state
    )

    model.eval()

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(
                torch.tensor(
                    X,
                    dtype=torch.float32,
                    device=DEVICE
                )
            )
        ).cpu().numpy()

    return np.abs(
        probabilities - 0.5
    ) * 2.0


def membership_inference_attack(
    model_state
):

    member_scores = np.concatenate(
        [
            confidence_scores(
                model_state,
                TRAIN_DATA[h][0]
            )
            for h in TRAIN_DATA
        ]
    )

    nonmember_scores = np.concatenate(
        [
            confidence_scores(
                model_state,
                TEST_DATA[h][0]
            )
            for h in TEST_DATA
        ]
    )

    n = min(
        len(member_scores),
        len(nonmember_scores)
    )

    rng = np.random.default_rng(
        SEED
    )

    member_scores = rng.choice(
        member_scores,
        n,
        replace=False
    )

    nonmember_scores = rng.choice(
        nonmember_scores,
        n,
        replace=False
    )

    scores = np.concatenate([
        member_scores,
        nonmember_scores
    ])

    labels = np.concatenate([
        np.ones(n),
        np.zeros(n)
    ])

    auc_value = roc_auc_score(
        labels,
        scores
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    youden = (
        tpr -
        fpr
    )

    valid = np.isfinite(
        thresholds
    )

    if valid.any():

        best_index = np.argmax(
            np.where(
                valid,
                youden,
                -np.inf
            )
        )

        threshold = float(
            thresholds[best_index]
        )

    else:

        threshold = 0.5

    predictions = (
        scores >= threshold
    ).astype(int)

    return {

        "MIA Accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "MIA AUC":
            auc_value,

        "MIA Threshold":
            threshold
    }


MIA_RESULTS = pd.DataFrame([

    {
        "Method":
            "Normal FL",
        **membership_inference_attack(
            NORMAL_STATE
        )
    },

    {
        "Method":
            "Fixed-DP FL",
        **membership_inference_attack(
            FIXED_STATE
        )
    },

    {
        "Method":
            "Proposed Adaptive-DP FL",
        **membership_inference_attack(
            ADAPTIVE_STATE
        )
    }

])

print("\n==============================")
print("MIA RESULTS")
print("==============================")

display(
    MIA_RESULTS.round(4)
)

# ============================================================
# 26. SAVE RESULTS
# ============================================================

FINAL_RESULTS.to_csv(
    OUT / "results" / "final_metrics.csv",
    index=False
)

HOSPITAL_RESULTS.to_csv(
    OUT / "results" / "hospital_results.csv",
    index=False
)

NOISE_TABLE.to_csv(
    OUT / "results" / "noise_table.csv",
    index=False
)

MIA_RESULTS.to_csv(
    OUT / "results" / "mia_results.csv",
    index=False
)

NORMAL_HISTORY.to_csv(
    OUT / "results" / "normal_history.csv",
    index=False
)

FIXED_HISTORY.to_csv(
    OUT / "results" / "fixed_dp_history.csv",
    index=False
)

ADAPTIVE_HISTORY.to_csv(
    OUT / "results" / "adaptive_dp_history.csv",
    index=False
)

# ============================================================
# 27. SAVE FINAL MODEL
# ============================================================

torch.save(
    ADAPTIVE_STATE,
    OUT /
    "model" /
    "Adaptive_DP_Global_Model.pt"
)

with open(
    OUT / "model" / "config.json",
    "w"
) as file:

    json.dump(
        {
            "title": TITLE,
            "features": FEATURES,
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "medians": MEDIANS,
            "ranges": RANGES,
            "cat_map": CAT_MAP,
            "hospital_config":
                HOSPITAL_CONFIG
        },
        file,
        indent=2
    )

# ============================================================
# 28. CREATE GRAPHS
# ============================================================

def save_training_plot(
    metric,
    filename,
    title
):

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        NORMAL_HISTORY["Round"],
        NORMAL_HISTORY[metric],
        marker="o",
        label="Normal FL"
    )

    plt.plot(
        FIXED_HISTORY["Round"],
        FIXED_HISTORY[metric],
        marker="o",
        label="Fixed-DP FL"
    )

    plt.plot(
        ADAPTIVE_HISTORY["Round"],
        ADAPTIVE_HISTORY[metric],
        marker="o",
        label="Proposed Adaptive-DP FL"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        metric
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "graphs" /
        filename,
        dpi=200
    )

    plt.close()


save_training_plot(
    "Accuracy",
    "accuracy_vs_round.png",
    "Accuracy Across Federated Rounds"
)

save_training_plot(
    "F1",
    "f1_vs_round.png",
    "F1 Score Across Federated Rounds"
)

# ------------------------------------------------------------
# Final metric graph
# ------------------------------------------------------------

ax = FINAL_RESULTS.set_index(
    "Method"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
].plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_title(
    "Final Model Comparison"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    OUT /
    "graphs" /
    "final_metric_comparison.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# MIA graph
# ------------------------------------------------------------

ax = MIA_RESULTS.set_index(
    "Method"
)[
    [
        "MIA Accuracy",
        "MIA AUC"
    ]
].plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.05
)

ax.set_title(
    "Membership Inference Attack Comparison"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    OUT /
    "graphs" /
    "mia_comparison.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# Missingness vs adaptive noise
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    NOISE_TABLE["Missing Level"] * 100,
    NOISE_TABLE["Adaptive Noise"],
    marker="o"
)

for _, row in NOISE_TABLE.iterrows():

    plt.annotate(
        row["Hospital"],
        (
            row["Missing Level"] * 100,
            row["Adaptive Noise"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel(
    "Simulated Missing Feature Level (%)"
)

plt.ylabel(
    "Adaptive Noise Multiplier"
)

plt.title(
    "Proposed Missingness-Driven Privacy Setting"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT /
    "graphs" /
    "missingness_vs_noise.png",
    dpi=200
)

plt.close()

# ============================================================
# 29. CONFUSION MATRICES
# ============================================================

for state, name, filename in [

    (
        NORMAL_STATE,
        "Normal FL",
        "cm_normal_fl.png"
    ),

    (
        FIXED_STATE,
        "Fixed-DP FL",
        "cm_fixed_dp_fl.png"
    ),

    (
        ADAPTIVE_STATE,
        "Adaptive-DP FL",
        "cm_adaptive_dp_fl.png"
    )

]:

    _, _, predictions = evaluate(
        state,
        *combined_data("test")
    )

    _, y_test = combined_data(
        "test"
    )

    cm = confusion_matrix(
        y_test.astype(int),
        predictions
    )

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(cm)

    plt.title(
        name +
        " - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Not CKD", "CKD"]
    )

    plt.yticks(
        [0, 1],
        ["Not CKD", "CKD"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "graphs" /
        filename,
        dpi=200
    )

    plt.close()

# ============================================================
# 30. PROJECT REPORT PDF
# ============================================================

styles = getSampleStyleSheet()

report_path = (
    OUT /
    "reports" /
    "project_report.pdf"
)

document = SimpleDocTemplate(
    str(report_path),
    pagesize=A4,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

story = []

story.append(
    Paragraph(
        TITLE,
        styles["Title"]
    )
)

story.append(
    Spacer(1, 12)
)

story.append(
    Paragraph(
        "CKD Case Study — Federated Learning with "
        "Hospital-wise Missing Features and Adaptive "
        "Privacy Protection.",
        styles["BodyText"]
    )
)

story.append(
    Spacer(1, 10)
)

story.append(
    Paragraph(
        "<b>Important:</b> Hospitals in this prototype "
        "are simulated clients created from public datasets. "
        "The missingness-to-noise mapping is a proposed "
        "experimental mechanism and is not presented as a "
        "clinically established rule.",
        styles["BodyText"]
    )
)

story.append(
    Spacer(1, 12)
)

# ------------------------------------------------------------
# Dataset section
# ------------------------------------------------------------

story.append(
    Paragraph(
        "1. Datasets",
        styles["Heading2"]
    )
)

story.append(
    Paragraph(
        "• UCI Chronic Kidney Disease Dataset — ID 336<br/>"
        "• Kaggle Chronic Kidney Failure Dataset Reboot<br/>"
        "• UCI Risk Factor Prediction of Chronic Kidney Disease — ID 857",
        styles["BodyText"]
    )
)

story.append(
    Spacer(1, 12)
)

# ------------------------------------------------------------
# Hospital configuration
# ------------------------------------------------------------

story.append(
    Paragraph(
        "2. Simulated Hospital Configuration",
        styles["Heading2"]
    )
)

hospital_table = [
    [
        "Hospital",
        "Dataset",
        "Missing",
        "Adaptive Noise"
    ]
]

dataset_names = {

    "Hospital 1":
        "UCI CKD 336",

    "Hospital 2":
        "Kaggle Reboot — Part A",

    "Hospital 3":
        "Kaggle Reboot — Part B",

    "Hospital 4":
        "UCI Risk Factor 857"
}

for hospital in HOSPITAL_CONFIG:

    missing = (
        HOSPITAL_CONFIG[
            hospital
        ]["missing"]
    )

    hospital_table.append(
        [
            hospital,
            dataset_names[hospital],
            f"{missing * 100:.0f}%",
            f"{adaptive_noise(missing):.2f}"
        ]
    )

table = Table(
    hospital_table,
    colWidths=[
        75,
        170,
        75,
        90
    ]
)

table.setStyle(
    TableStyle([
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        )
    ])
)

story.append(table)

story.append(
    Spacer(1, 12)
)

# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

story.append(
    Paragraph(
        "3. Final Model Performance",
        styles["Heading2"]
    )
)

performance_table = [
    [
        "Method",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
]

for _, row in FINAL_RESULTS.iterrows():

    performance_table.append(
        [
            row["Method"],
            f'{row["Accuracy"]:.4f}',
            f'{row["Precision"]:.4f}',
            f'{row["Recall"]:.4f}',
            f'{row["F1"]:.4f}',
            f'{row["ROC-AUC"]:.4f}'
        ]
    )

table = Table(
    performance_table,
    colWidths=[
        130,
        65,
        65,
        65,
        65,
        65
    ]
)

table.setStyle(
    TableStyle([
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        )
    ])
)

story.append(table)

story.append(
    Spacer(1, 12)
)

# ------------------------------------------------------------
# MIA
# ------------------------------------------------------------

story.append(
    Paragraph(
        "4. Privacy Evaluation — "
        "Confidence-based Membership Inference Attack",
        styles["Heading2"]
    )
)

mia_table = [
    [
        "Method",
        "MIA Accuracy",
        "MIA AUC"
    ]
]

for _, row in MIA_RESULTS.iterrows():

    mia_table.append(
        [
            row["Method"],
            f'{row["MIA Accuracy"]:.4f}',
            f'{row["MIA AUC"]:.4f}'
        ]
    )

table = Table(
    mia_table,
    colWidths=[
        180,
        100,
        100
    ]
)

table.setStyle(
    TableStyle([
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.grey
        ),

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        )
    ])
)

story.append(table)

story.append(
    Spacer(1, 12)
)

# ------------------------------------------------------------
# Graphs
# ------------------------------------------------------------

story.append(
    Paragraph(
        "5. Graphical Results",
        styles["Heading2"]
    )
)

graph_files = [

    "accuracy_vs_round.png",

    "f1_vs_round.png",

    "final_metric_comparison.png",

    "mia_comparison.png",

    "missingness_vs_noise.png",

    "cm_adaptive_dp_fl.png"
]

for graph in graph_files:

    path = (
        OUT /
        "graphs" /
        graph
    )

    if path.exists():

        story.append(
            RLImage(
                str(path),
                width=6.3 * inch,
                height=3.5 * inch
            )
        )

        story.append(
            Spacer(1, 8)
        )

# ------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------

story.append(
    Paragraph(
        "6. Conclusion",
        styles["Heading2"]
    )
)

story.append(
    Paragraph(
        "This prototype demonstrates a privacy-preserving "
        "federated learning workflow for multi-hospital "
        "medical data analysis using CKD as the case study. "
        "Raw patient records remain within simulated hospital "
        "clients. Hospital-wise missing-feature levels are "
        "assessed and used by the proposed adaptive mechanism "
        "to select a privacy-noise setting. Protected local "
        "model updates are aggregated using Federated Averaging "
        "(FedAvg). Model performance is evaluated using "
        "Accuracy, Precision, Recall, F1-score and ROC-AUC, "
        "while privacy leakage is explored using a "
        "confidence-based membership inference attack prototype.",
        styles["BodyText"]
    )
)

document.build(
    story
)

# ============================================================
# 31. CREATE README
# ============================================================

readme_text = f"""
# {TITLE}

## CKD Case Study

This project implements:

- Multi-hospital Federated Learning
- Hospital-wise missing clinical features
- Adaptive privacy-noise selection
- Differential-privacy-style update protection
- FedAvg aggregation
- CKD prediction
- Membership Inference Attack evaluation
- Interactive deployment using Hugging Face Spaces

## Simulated Hospitals

Hospital 1: UCI CKD 336 — 5% missingness

Hospital 2: Kaggle CKD Reboot Part A — 15% missingness

Hospital 3: Kaggle CKD Reboot Part B — 25% missingness

Hospital 4: UCI Risk Factor CKD 857 — 40% missingness

## Adaptive Privacy Mapping

5%  -> 0.05 noise

15% -> 0.10 noise

25% -> 0.20 noise

40% -> 0.30 noise

The mapping is a proposed experimental mechanism.

## Important

The hospitals are simulated from public datasets and do not represent real hospital-to-hospital data sharing.
"""

with open(
    OUT / "README.md",
    "w"
) as file:

    file.write(
        readme_text
    )

# ============================================================
# 32. CREATE REQUIREMENTS.TXT
# ============================================================

requirements = """gradio
torch
pandas
numpy
matplotlib
scikit-learn
reportlab
"""

with open(
    OUT / "requirements.txt",
    "w"
) as file:

    file.write(
        requirements
    )

# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nProject output folder:"
)

print(OUT)

print(
    "\nFiles created:"
)

for path in sorted(
    OUT.rglob("*")
):

    if path.is_file():

        print(
            " -",
            path.relative_to(OUT)
        )

print(
    "\nFinal Adaptive-DP result:"
)

display(
    FINAL_RESULTS.round(4)
)

print(
    "\nMIA result:"
)

display(
    MIA_RESULTS.round(4)
)

print(
    "\nProject PDF:"
)

print(
    OUT /
    "reports" /
    "project_report.pdf"
)

print("\nNEXT STEP:")
print(
    "Download the CKD_Federated_Project_Output folder "
    "and upload its files to your Hugging Face Space."
)