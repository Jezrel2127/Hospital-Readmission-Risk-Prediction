import pandas as pd
import numpy as np

def detect_noise(data):

    df = data.copy()

    results = []

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    excluded_columns = [
        "encounter_id",
        "patient_nbr"
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        negative_count = int(
            (values < 0).sum()
        )

        # Check infinite values
        infinite_count = int(
            np.isinf(values).sum()
        )
        valid_values = values.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(valid_values) > 0:

            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)

            iqr = q3 - q1

            lower_limit = q1 - (1.5 * iqr)
            upper_limit = q3 + (1.5 * iqr)

            unusual_count = int(
                (
                    (valid_values < lower_limit) |
                    (valid_values > upper_limit)
                ).sum()
            )

        else:

            lower_limit = np.nan
            upper_limit = np.nan
            unusual_count = 0

        results.append({
            "Column": column,
            "Negative Values": negative_count,
            "Infinite Values": infinite_count,
            "Unusual Values (IQR)": unusual_count,
            "Lower Limit": (
                round(lower_limit, 2)
                if pd.notna(lower_limit)
                else None
            ),
            "Upper Limit": (
                round(upper_limit, 2)
                if pd.notna(upper_limit)
                else None
            )
        })

    results_df = pd.DataFrame(results)

    summary = {
        "total_columns_checked": len(numeric_columns),
        "total_negative_values": (
            int(results_df["Negative Values"].sum())
            if not results_df.empty else 0
        ),
        "total_infinite_values": (
            int(results_df["Infinite Values"].sum())
            if not results_df.empty else 0
        ),
        "total_unusual_values": (
            int(results_df["Unusual Values (IQR)"].sum())
            if not results_df.empty else 0
        )
    }

    return results_df, summary
