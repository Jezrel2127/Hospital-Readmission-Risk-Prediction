import pandas as pd
import numpy as np

def preprocess_data(data):

    df = data.copy()

    if "readmitted" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'readmitted' column."
        )
    df = df.dropna(subset=["readmitted"]).copy()
    identifier_columns = [
        "encounter_id",
        "patient_nbr"
    ]

    df = df.drop(
        columns=[
            col for col in identifier_columns
            if col in df.columns
        ]
    )
    y = (df["readmitted"] == "<30").astype(int)

    X = df.drop(columns=["readmitted"])
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    coded_categorical_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]

    for column in coded_categorical_columns:
        if column in X.columns:
            X[column] = X[column].astype("string")

            if column not in categorical_columns:
                categorical_columns.append(column)
                
    numerical_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numerical_columns:
        X[column] = X[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

        if X[column].isnull().any():
            X[column] = X[column].fillna(
                X[column].median()
            )
    for column in categorical_columns:
        if X[column].isnull().any():
            modes = X[column].mode()

            fill_value = (
                modes.iloc[0]
                if not modes.empty
                else "Unknown"
            )

            X[column] = X[column].fillna(fill_value)
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int
    )

    X_encoded = X_encoded.replace(
        [np.inf, -np.inf],
        np.nan
    )
    encoded_medians = X_encoded.median().fillna(0)
    scaling_minimum = X_encoded.min().fillna(0)
    scaling_maximum = X_encoded.max().fillna(0)
    
    X_scaled = X_encoded.copy()

    for column in X_scaled.columns:

        minimum = scaling_minimum[column]
        maximum = scaling_maximum[column]

        if maximum != minimum:
            X_scaled[column] = (
                (X_scaled[column] - minimum)
                / (maximum - minimum)
            )
        else:
            X_scaled[column] = 0

    scaling_results = pd.DataFrame([
        {
            "Column": column,
            "Minimum": scaling_minimum[column],
            "Maximum": scaling_maximum[column],
            "Scaling": (
                "Min-Max"
                if scaling_maximum[column] != scaling_minimum[column]
                else "Constant"
            )
        }
        for column in X_encoded.columns
    ])

    summary = {
        "original_features": len(X.columns),
        "encoded_features": len(X_encoded.columns),
        "categorical_columns": categorical_columns,
        "scaled_features": len(X_scaled.columns),
        "scaling_results": scaling_results,
        "feature_columns": X_encoded.columns.tolist(),
        "encoded_medians": encoded_medians,
        "scaling_minimum": scaling_minimum,
        "scaling_maximum": scaling_maximum,
        "coded_categorical_columns": coded_categorical_columns
    }

    return X_scaled, y, summary
