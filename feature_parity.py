# ===========================
# CI/CD Train ↔ Inference Feature Parity Validation
# ===========================

import json
from pathlib import Path
import pandas as pd


def validate_inference_schema(X: pd.DataFrame, schema_path: Path) -> None:
    """
    Validates that inference features match the training schema.

    Enforces:
      - All required columns exist
      - Column order is preserved
      - Dtypes match training schema (if recorded)
    """

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        saved_schema = json.load(f)

    schema_cols = saved_schema.get("features")
    if not schema_cols:
        raise KeyError("Schema file missing 'features' key")

    # Column existence
    missing_cols = set(schema_cols) - set(X.columns)
    extra_cols = set(X.columns) - set(schema_cols)
    if missing_cols:
        raise ValueError(f"Missing columns in inference data: {missing_cols}")
    if extra_cols:
        raise ValueError(f"Unexpected extra columns in inference data: {extra_cols}")

    # Column order
    if list(X.columns) != list(schema_cols):
        raise ValueError(
            "Column order mismatch between inference features and training schema.\n"
            f"Expected: {schema_cols}\n"
            f"Got:      {list(X.columns)}"
        )

    # Dtype validation (best-effort)
    saved_dtypes = saved_schema.get("dtypes", {})
    dtype_errors = []
    for col in schema_cols:
        expected_dtype = saved_dtypes.get(col)
        if expected_dtype is not None:
            actual_dtype = str(X[col].dtype)
            if actual_dtype != expected_dtype:
                dtype_errors.append((col, expected_dtype, actual_dtype))

    if dtype_errors:
        formatted = ", ".join([f"{c}: expected={e}, got={a}" for c, e, a in dtype_errors])
        raise TypeError(f"Dtype mismatches detected: {formatted}")

    print("✅ Train ↔ inference feature parity validated")