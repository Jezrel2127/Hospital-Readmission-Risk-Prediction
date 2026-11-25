import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from models.preprocessing import preprocess_data
from models.train_model import train_models, prepare_new_patient
from models.noise_detection import detect_noise


st.set_page_config(
    page_title="Hospital Readmission Risk Prediction",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #173F75;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #EAF3FF;
}

[data-testid="stSidebar"] * {
    color: #173F75;
}

h1, h2, h3, h4 {
    color: #173F75 !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #26364A;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #1769C2;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #0F4F96;
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background-color: #F4F8FF;
    border: 1px solid #D6E5F7;
    padding: 16px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #173F75 !important;
}

[data-testid="stMetricValue"] {
    color: #1769C2 !important;
}

[data-testid="stAlert"] {
    background-color: #1769C2 !important;
    border: 1px solid #0F4F96 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] *,
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] {
    background-color: #F4F8FF;
    border: 1px dashed #8CB8E8;
    border-radius: 10px;
    padding: 12px;
}

hr {
    border-color: #D6E5F7;
}
</style>
""", unsafe_allow_html=True)


state_defaults = {
    "original_df": None,
    "sample_df": None,
    "cleaned_df": None,
    "preprocessed_X": None,
    "preprocessed_y": None,
    "preprocessing_summary": None,
    "model_results": None,
    "trained_models": None,
    "X_test": None,
    "y_test": None,
    "uploaded_signature": None,
    "prediction_results": None,
    "prediction_patient": None,
    "noise_results": None,
    "noise_summary": None
}

for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
        
def reset_model_results():

    st.session_state.model_results = None
    st.session_state.trained_models = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.prediction_results = None
    st.session_state.prediction_patient = None


def reset_preprocessing():

    st.session_state.preprocessed_X = None
    st.session_state.preprocessed_y = None
    st.session_state.preprocessing_summary = None

    reset_model_results()


def reset_noise_results():

    st.session_state.noise_results = None
    st.session_state.noise_summary = None


def show_header():

    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="color:#D62828; font-size:42px; font-weight:bold;">✚</span>
            <span style="color:#173F75; font-size:30px; font-weight:800;">
                Hospital Readmission Risk Prediction
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_stratified_sample(data, sample_size=5000):

    if "readmitted" not in data.columns:
        raise ValueError(
            "The dataset must contain a readmitted column."
        )

    if len(data) <= sample_size:
        return data.copy().reset_index(drop=True)

    grouped = data.groupby(
        "readmitted",
        dropna=False
    )

    proportions = grouped.size() / len(data)
    exact_allocations = proportions * sample_size

    allocations = exact_allocations.apply(int)

    remaining = sample_size - int(allocations.sum())

    fractions = exact_allocations - allocations

    for category in fractions.sort_values(
        ascending=False
    ).index[:remaining]:

        allocations.loc[category] += 1

    sampled_groups = []

    for category, group in grouped:

        number = int(allocations.loc[category])

        if number > 0:

            sampled_groups.append(
                group.sample(
                    n=number,
                    random_state=42
                )
            )

    result = pd.concat(sampled_groups)

    return result.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


def get_numeric_columns(data):

    excluded = [
        "encounter_id",
        "patient_nbr"
    ]

    return [
        column
        for column in data.select_dtypes(
            include="number"
        ).columns
        if column not in excluded
    ]


def build_patient_input(source_data, patient_input):

    patient_base = {}

    for column in source_data.columns:

        if column in [
            "readmitted",
            "encounter_id",
            "patient_nbr"
        ]:
            continue

        if pd.api.types.is_numeric_dtype(source_data[column]):

            median_value = source_data[column].median()

            patient_base[column] = (
                median_value
                if pd.notna(median_value)
                else 0
            )

        else:

            modes = source_data[column].mode()

            patient_base[column] = (
                modes.iloc[0]
                if not modes.empty
                else "Unknown"
            )

    patient_base.update(patient_input)

    return pd.DataFrame([patient_base])


def prepare_patient_features(patient_df, summary):

    patient_df = patient_df.copy()

    patient_df = patient_df.drop(
        columns=[
            "readmitted",
            "encounter_id",
            "patient_nbr"
        ],
        errors="ignore"
    )

    categorical_columns = patient_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    coded_categorical_columns = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]

    for column in coded_categorical_columns:

        if column in patient_df.columns:

            patient_df[column] = patient_df[column].astype("string")

            if column not in categorical_columns:
                categorical_columns.append(column)

    patient_encoded = pd.get_dummies(
        patient_df,
        columns=categorical_columns,
        dtype=int
    )

    feature_columns = summary["feature_columns"]

    patient_encoded = patient_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    patient_encoded = patient_encoded.replace(
        [np.inf, -np.inf],
        np.nan
    )

    patient_encoded = patient_encoded.fillna(
        summary["encoded_medians"]
    )

    minimum = summary["scaling_minimum"]
    maximum = summary["scaling_maximum"]

    denominator = (maximum - minimum).replace(0, 1)

    patient_scaled = (
        patient_encoded - minimum
    ) / denominator

    patient_scaled = patient_scaled.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return patient_scaled

st.sidebar.title("HOSPITAL SYSTEM")
st.sidebar.caption("DATA MINING PROJECT")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Dataset Loading",
        "Data Quality Analysis",
        "Noise Data Detection",
        "Data Cleaning",
        "Data Preprocessing",
        "Data Visualization",
        "Model Development",
        "Model Evaluation",
        "Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Hospital Readmission Risk Prediction"
)

st.sidebar.caption(
    "Data Mining Project"
)

show_header()
st.markdown("---")

if page == "Dashboard":

    st.title("Dashboard")

    st.write(
        "Welcome to the Hospital Readmission Risk Prediction System."
    )

    st.info(
        "Upload your hospital dataset and generate a sample to begin."
    )

    if st.session_state.original_df is not None:

        original = st.session_state.original_df

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Original Records",
            f"{len(original):,}"
        )

        col2.metric(
            "Total Columns",
            len(original.columns)
        )

        sample_count = (
            len(st.session_state.sample_df)
            if st.session_state.sample_df is not None
            else 0
        )

        col3.metric(
            "Sample Records",
            f"{sample_count:,}"
        )

    else:

        st.warning(
            "No dataset uploaded yet. Go to Dataset Loading."
        )

    st.markdown("---")

    st.subheader("System Modules")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        - **Dataset Loading**
          - Upload and preview the hospital dataset.

        - **Data Quality Analysis**
          - Check missing values, duplicate records, constant columns, and numerical outliers.

        - **Noise Data Detection**
          - Identify negative, infinite, and unusual numerical values.

        - **Data Cleaning**
          - Remove duplicate records and handle missing values.

        - **Data Preprocessing**
          - Encode categorical data and scale numerical features.
        """)

    with col2:

        st.markdown("""
        - **Data Visualization**
          - View charts and distributions from the dataset.

        - **Model Development**
          - Train Decision Tree and K-Nearest Neighbors algorithms.

        - **Model Evaluation**
          - Compare model performance using evaluation metrics.

        - **Prediction**
          - Generate experimental readmission predictions using trained models.
        """)

    st.markdown("---")

    st.subheader("Project Progress")

    progress_items = {
        "Dataset Loading": st.session_state.original_df is not None,
        "Dataset Sampling": st.session_state.sample_df is not None,
        "Data Quality Analysis": st.session_state.sample_df is not None,
        "Noise Data Detection": st.session_state.noise_results is not None,
        "Data Cleaning": st.session_state.cleaned_df is not None,
        "Data Preprocessing": st.session_state.preprocessed_X is not None,
        "Model Development": st.session_state.trained_models is not None,
        "Model Evaluation": st.session_state.model_results is not None,
        "Prediction": st.session_state.prediction_results is not None
    }

    for item, completed in progress_items.items():

        if completed:
            st.success(f"{item} - Completed")
        else:
            st.info(f"{item} - Pending")

elif page == "Dataset Loading":

    st.title("Dataset Loading")

    st.write(
        "Upload the hospital dataset in CSV format."
    )

    uploaded_file = st.file_uploader(
        "Choose your hospital dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            file_signature = (
                uploaded_file.name,
                uploaded_file.size
            )

            if file_signature != st.session_state.uploaded_signature:

                data = pd.read_csv(
                    uploaded_file,
                    na_values=["?"],
                    keep_default_na=True,
                    low_memory=False
                )

                st.session_state.original_df = data
                st.session_state.sample_df = None
                st.session_state.cleaned_df = None

                reset_preprocessing()
                reset_noise_results()

                st.session_state.uploaded_signature = file_signature

            data = st.session_state.original_df

            st.success(
                "Dataset successfully uploaded!"
            )

            st.markdown("---")
            st.subheader("Dataset Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Records",
                f"{len(data):,}"
            )

            col2.metric(
                "Total Columns",
                len(data.columns)
            )

            col3.metric(
                "Missing Values",
                f"{int(data.isnull().sum().sum()):,}"
            )

            st.markdown("---")
            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("Generate Dataset Sample")

            st.write(
                "The system will generate a proportional sample of up to 5,000 records based on the readmission categories."
            )

            if st.button(
                "Generate 5,000-Record Sample",
                use_container_width=True
            ):

                sample = generate_stratified_sample(
                    data,
                    sample_size=5000
                )

                st.session_state.sample_df = sample
                st.session_state.cleaned_df = None

                reset_preprocessing()
                reset_noise_results()

                st.success(
                    f"Sample successfully generated! Total records: {len(sample):,}"
                )

            if st.session_state.sample_df is not None:

                sample = st.session_state.sample_df

                st.markdown("---")
                st.subheader("Generated Sample Preview")

                st.metric(
                    "Sample Size",
                    f"{len(sample):,}"
                )

                st.dataframe(
                    sample.head(10),
                    use_container_width=True
                )

                st.subheader("Readmission Distribution")

                distribution = sample[
                    "readmitted"
                ].value_counts()

                st.bar_chart(distribution)

                st.download_button(
                    "Download Sample Dataset",
                    data=sample.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="hospital_sample_5000.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Unable to load dataset: {error}"
            )
elif page == "Data Quality Analysis":

    st.title("Data Quality Analysis")

    if st.session_state.sample_df is None:

        st.warning(
            "Please load the dataset and generate a sample first."
        )

    else:

        data = st.session_state.sample_df

        st.subheader("1. Missing Values Analysis")

        missing_count = data.isnull().sum()
        missing_percentage = data.isnull().mean() * 100

        missing_table = pd.DataFrame({
            "Column": data.columns,
            "Missing Values": missing_count.values,
            "Missing Percentage (%)": missing_percentage.values
        })

        missing_table = missing_table[
            missing_table["Missing Values"] > 0
        ].sort_values(
            by="Missing Values",
            ascending=False
        )

        if missing_table.empty:

            st.success("No missing values detected.")

        else:

            st.dataframe(
                missing_table,
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("2. Duplicate Data Analysis")

        duplicate_count = int(
            data.duplicated().sum()
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Duplicate Rows",
            f"{duplicate_count:,}"
        )

        col2.metric(
            "Unique Rows",
            f"{len(data.drop_duplicates()):,}"
        )

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count:,} duplicate rows detected."
            )

        else:

            st.success("No duplicate rows detected.")

        st.markdown("---")
        st.subheader("3. Constant Column Analysis")

        constant_columns = [
            column
            for column in data.columns
            if data[column].nunique(dropna=False) <= 1
        ]

        if constant_columns:

            st.warning(
                "Constant columns detected: " +
                ", ".join(constant_columns)
            )

        else:

            st.success("No constant columns detected.")

        st.markdown("---")
        st.subheader("4. Outlier Detection Using IQR")

        numeric_columns = get_numeric_columns(data)

        if numeric_columns:

            outlier_results = []

            for column in numeric_columns:

                values = data[column].dropna()

                if len(values) == 0:
                    continue

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                lower_limit = q1 - (1.5 * iqr)
                upper_limit = q3 + (1.5 * iqr)

                outliers = values[
                    (values < lower_limit) |
                    (values > upper_limit)
                ]

                outlier_results.append({
                    "Column": column,
                    "Outlier Count": len(outliers),
                    "Lower Limit": round(lower_limit, 2),
                    "Upper Limit": round(upper_limit, 2)
                })

            st.dataframe(
                pd.DataFrame(outlier_results),
                use_container_width=True
            )

        else:

            st.info(
                "No numerical columns available."
            )

        st.markdown("---")
        st.subheader("5. Mean, Median, and Mode")

        if numeric_columns:

            summary_data = []

            for column in numeric_columns:

                values = data[column].dropna()

                if len(values) == 0:
                    continue

                modes = values.mode()

                mode_value = (
                    modes.iloc[0]
                    if not modes.empty
                    else None
                )

                summary_data.append({
                    "Column": column,
                    "Mean": round(values.mean(), 2),
                    "Median": round(values.median(), 2),
                    "Mode": mode_value
                })

            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True
            )

elif page == "Noise Data Detection":

    st.title("Noise Data Detection")

    st.write(
        "Identify potentially noisy or unusual values in the hospital dataset."
    )

    st.markdown("---")

    if st.session_state.sample_df is None:

        st.warning(
            "Please upload the dataset and generate a sample first."
        )

    else:

        data = st.session_state.sample_df

        st.subheader("Detection Methods")

        st.markdown("""
        - **Negative Value Detection:** Identifies negative values in numerical columns.
        - **Infinite Value Detection:** Identifies positive or negative infinite values.
        - **IQR-Based Detection:** Flags numerical values outside the Interquartile Range limits.
        """)

        st.info(
            "IQR outliers are potential unusual values. They are not automatically considered errors or removed."
        )

        if st.button(
            "Detect Noise Data",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing numerical columns..."
                ):

                    noise_results, noise_summary = detect_noise(data)

                    st.session_state.noise_results = noise_results
                    st.session_state.noise_summary = noise_summary

                st.success(
                    "Noise data detection completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Noise detection failed: {error}"
                )

        if st.session_state.noise_results is not None:

            noise_results = st.session_state.noise_results
            noise_summary = st.session_state.noise_summary

            st.markdown("---")
            st.subheader("Noise Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Columns Checked",
                noise_summary["total_columns_checked"]
            )

            col2.metric(
                "Negative Values",
                f"{noise_summary['total_negative_values']:,}"
            )

            col3.metric(
                "Infinite Values",
                f"{noise_summary['total_infinite_values']:,}"
            )

            col4.metric(
                "Unusual Values",
                f"{noise_summary['total_unusual_values']:,}"
            )

            st.markdown("---")
            st.subheader("Detailed Noise Detection Results")

            st.dataframe(
                noise_results,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")
            st.subheader("Noise Detection Graph")

            graph_data = noise_results.set_index(
                "Column"
            )[
                [
                    "Negative Values",
                    "Infinite Values",
                    "Unusual Values (IQR)"
                ]
            ]

            st.bar_chart(graph_data)

            st.markdown("---")

            st.download_button(
                "Download Noise Detection Report",
                data=noise_results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="noise_detection_report.csv",
                mime="text/csv",
                use_container_width=True
            )

elif page == "Data Cleaning":

    st.title("Data Cleaning")

    if st.session_state.sample_df is None:

        st.warning(
            "Please generate a 5,000-record sample first."
        )

    else:

        data = st.session_state.sample_df.copy()

        st.write(
            "Remove duplicate rows and handle missing values."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Original Rows",
            f"{len(data):,}"
        )

        col2.metric(
            "Missing Values",
            f"{int(data.isnull().sum().sum()):,}"
        )

        if st.button(
            "Clean Dataset",
            use_container_width=True
        ):

            cleaned = data.drop_duplicates().copy()

            numeric_columns = cleaned.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_columns = cleaned.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            for column in numeric_columns:

                if cleaned[column].isnull().any():

                    median_value = cleaned[column].median()

                    cleaned[column] = cleaned[column].fillna(
                        median_value
                    )

            for column in categorical_columns:

                if cleaned[column].isnull().any():

                    modes = cleaned[column].mode()

                    if not modes.empty:

                        cleaned[column] = cleaned[column].fillna(
                            modes.iloc[0]
                        )

                    else:

                        cleaned[column] = cleaned[column].fillna(
                            "Unknown"
                        )

            st.session_state.cleaned_df = cleaned

            reset_preprocessing()

            st.success(
                "Data cleaning completed successfully!"
            )

        if st.session_state.cleaned_df is not None:

            cleaned = st.session_state.cleaned_df

            st.markdown("---")
            st.subheader("Cleaning Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Original Rows",
                f"{len(data):,}"
            )

            col2.metric(
                "Cleaned Rows",
                f"{len(cleaned):,}"
            )

            col3.metric(
                "Remaining Missing Values",
                f"{int(cleaned.isnull().sum().sum()):,}"
            )

            st.subheader("Cleaned Dataset Preview")

            st.dataframe(
                cleaned.head(10),
                use_container_width=True
            )

            st.download_button(
                "Download Cleaned Dataset",
                data=cleaned.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="hospital_cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

elif page == "Data Preprocessing":

    st.title("Data Preprocessing")

    st.write(
        "Prepare the hospital dataset for machine learning."
    )

    st.markdown("---")

    if (
        st.session_state.sample_df is None
        and st.session_state.cleaned_df is None
    ):

        st.warning(
            "Please upload your dataset and generate a sample first."
        )

    else:

        if st.session_state.cleaned_df is not None:

            data = st.session_state.cleaned_df.copy()

            st.info(
                "Using the cleaned dataset for preprocessing."
            )

        else:

            data = st.session_state.sample_df.copy()

            st.info(
                "Using the generated sample for preprocessing."
            )

        st.subheader("Preprocessing Steps")

        st.markdown("""
        - **Categorical Encoding**
          - Converts categorical values into numerical columns.

        - **Missing Value Handling**
          - Handles missing numerical and categorical values.

        - **Feature Scaling**
          - Applies Min-Max Scaling to numerical features.

        - **Target Preparation**
          - Converts readmission status into a binary target.
        """)

        col1, col2 = st.columns(2)

        col1.metric(
            "Input Records",
            f"{len(data):,}"
        )

        col2.metric(
            "Input Columns",
            len(data.columns)
        )

        st.markdown("---")

        if st.button(
            "Run Data Preprocessing",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing dataset. Please wait..."
                ):

                    X_processed, y_processed, summary = preprocess_data(
                        data
                    )

                    st.session_state.preprocessed_X = X_processed
                    st.session_state.preprocessed_y = y_processed
                    st.session_state.preprocessing_summary = summary

                    reset_model_results()

                st.success(
                    "Data preprocessing completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Preprocessing error: {error}"
                )

        if st.session_state.preprocessed_X is not None:

            X_processed = st.session_state.preprocessed_X
            y_processed = st.session_state.preprocessed_y
            summary = st.session_state.preprocessing_summary

            st.markdown("---")
            st.subheader("Preprocessing Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Input Features",
                summary["original_features"]
            )

            col2.metric(
                "Encoded Features",
                summary["encoded_features"]
            )

            col3.metric(
                "Scaled Features",
                summary["scaled_features"]
            )

            st.markdown("---")
            st.subheader("Categorical Encoding Results")

            st.write(
                f"Categorical columns processed: {len(summary['categorical_columns'])}"
            )

            st.dataframe(
                pd.DataFrame({
                    "Categorical Columns": summary["categorical_columns"]
                }),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("Feature Scaling Results")

            st.dataframe(
                summary["scaling_results"],
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("Processed Dataset Preview")

            st.dataframe(
                X_processed.head(10),
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("Target Variable Preview")

            target_preview = pd.DataFrame({
                "Readmission Target": y_processed
            })

            st.dataframe(
                target_preview.head(10),
                use_container_width=True
            )

            st.subheader("Target Distribution")

            target_distribution = y_processed.value_counts().rename(
                index={
                    0: "Not readmitted within 30 days",
                    1: "Readmitted within 30 days"
                }
            )

            st.bar_chart(
                target_distribution
            )

            st.markdown("---")
            st.subheader("Download Preprocessed Data")

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "Download Processed Features",
                    data=X_processed.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="hospital_preprocessed_features.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:

                st.download_button(
                    "Download Target Variable",
                    data=target_preview.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="hospital_target_variable.csv",
                    mime="text/csv",
                    use_container_width=True
                )

elif page == "Data Visualization":

    st.title("Data Visualization")

    if st.session_state.sample_df is None:

        st.warning(
            "Please generate a 5,000-record sample first."
        )

    else:

        data = st.session_state.sample_df

        st.write(
            "Explore the hospital dataset through charts and visual summaries."
        )

        st.markdown("---")
        st.subheader("1. Readmission Distribution")

        if "readmitted" in data.columns:

            st.bar_chart(
                data["readmitted"].value_counts()
            )

        st.markdown("---")
        st.subheader("2. Patient Age Distribution")

        if "age" in data.columns:

            st.bar_chart(
                data["age"].value_counts()
            )

        st.markdown("---")
        st.subheader("3. Time Spent in Hospital")

        if "time_in_hospital" in data.columns:

            st.bar_chart(
                data["time_in_hospital"].value_counts().sort_index()
            )

        st.markdown("---")
        st.subheader("4. Number of Medications")

        if "num_medications" in data.columns:

            st.line_chart(
                data["num_medications"].value_counts().sort_index()
            )

elif page == "Model Development":

    st.title("Model Development")

    st.write(
        "Train Decision Tree and K-Nearest Neighbors models "
        "using the preprocessed hospital dataset."
    )

    st.markdown("---")

    if st.session_state.preprocessed_X is None:

        st.warning(
            "Please complete Data Preprocessing before training the models."
        )

    else:

        X = st.session_state.preprocessed_X
        y = st.session_state.preprocessed_y

        st.subheader("Training Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(X):,}"
        )

        col2.metric(
            "Total Features",
            len(X.columns)
        )

        col3.metric(
            "Target Classes",
            y.nunique()
        )

        st.markdown("---")

        st.subheader("Machine Learning Algorithms")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Decision Tree")

            st.write(
                "Uses Gini Impurity to select the best feature "
                "and divide the dataset into decision branches."
            )

        with col2:

            st.markdown("### K-Nearest Neighbors")

            st.write(
                "Classifies records using the distance "
                "between a patient and nearby training records."
            )

        st.markdown("---")

        st.subheader("Training Configuration")

        col1, col2, col3 = st.columns(3)

        col1.metric("Training Set", "80%")
        col2.metric("Testing Set", "20%")
        col3.metric("KNN Neighbors", "5")

        st.caption(
            "Decision Tree maximum depth: 5"
        )

        if st.button(
            "Train Machine Learning Models",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Training Decision Tree and KNN. Please wait..."
                ):

                    results, trained_models, X_test, y_test = train_models(
                        X,
                        y
                    )

                    st.session_state.model_results = results
                    st.session_state.trained_models = trained_models
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    st.session_state.prediction_results = None
                    st.session_state.prediction_patient = None

                st.success(
                    "Both machine learning models were trained successfully!"
                )

            except Exception as error:

                st.error(
                    f"Model training failed: {error}"
                )

        if st.session_state.trained_models is not None:

            st.markdown("---")
            st.subheader("Training Status")

            st.success(
                "Decision Tree model trained and saved."
            )

            st.success(
                "K-Nearest Neighbors model trained and saved."
            )

            st.info(
                "The trained model files are saved inside the models folder."
            )

            st.code(
                "models/decision_tree_model.pkl\n"
                "models/knn_model.pkl"
            )

            st.info(
                "Go to Model Evaluation to view the performance results."
            )

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.write(
        "Evaluate and compare the performance of the trained models."
    )

    st.markdown("---")

    if st.session_state.model_results is None:

        st.warning(
            "No evaluation results available. "
            "Please train the models first."
        )

    else:

        results = st.session_state.model_results

        for model_name, metrics in results.items():

            st.subheader(model_name)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{metrics['Accuracy'] * 100:.2f}%"
            )

            col2.metric(
                "Precision",
                f"{metrics['Precision'] * 100:.2f}%"
            )

            col3.metric(
                "Recall",
                f"{metrics['Recall'] * 100:.2f}%"
            )

            col4.metric(
                "F1-Score",
                f"{metrics['F1-Score'] * 100:.2f}%"
            )

            st.markdown("---")
            st.write("Confusion Matrix")

            confusion = pd.DataFrame(
                metrics["Confusion Matrix"],
                index=[
                    "Actual: Not Readmitted",
                    "Actual: Readmitted"
                ],
                columns=[
                    "Predicted: Not Readmitted",
                    "Predicted: Readmitted"
                ]
            )

            st.dataframe(
                confusion,
                use_container_width=True
            )

            st.write("Classification Report")

            st.code(
                metrics["Classification Report"]
            )

            st.markdown("---")

        st.subheader("Model Performance Comparison")

        comparison = pd.DataFrame({
            model_name: {
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"]
            }
            for model_name, metrics in results.items()
        })

        st.write("Performance Metrics (%)")

        st.dataframe(
            (comparison * 100).round(2),
            use_container_width=True
        )

        st.subheader("Performance Graph")

        st.bar_chart(
            comparison.T * 100
        )

        best_model = max(
            results,
            key=lambda name: results[name]["F1-Score"]
        )

        st.success(
            f"Best model based on F1-Score: {best_model}"
        )

        st.caption(
            "Evaluation uses the held-out 20% testing dataset. "
            "These experimental results do not establish clinical suitability."
        )

elif page == "Prediction":

    st.title("Readmission Risk Prediction")

    st.write(
        "Enter patient information to generate experimental "
        "readmission predictions using Decision Tree and KNN."
    )

    st.markdown("---")

    if st.session_state.trained_models is None:

        st.warning(
            "The machine learning models have not yet been trained."
        )

        st.info(
            "Complete Data Preprocessing and Model Development first."
        )

    elif st.session_state.preprocessing_summary is None:

        st.warning(
            "Preprocessing information is unavailable. "
            "Please run Data Preprocessing again."
        )

    elif (
        st.session_state.cleaned_df is None
        and st.session_state.sample_df is None
    ):

        st.warning(
            "The original patient dataset is unavailable."
        )

    else:

        if st.session_state.cleaned_df is not None:
            source_data = st.session_state.cleaned_df.copy()
        else:
            source_data = st.session_state.sample_df.copy()

        summary = st.session_state.preprocessing_summary

        required_columns = [
            "age",
            "time_in_hospital",
            "num_lab_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "diag_1",
            "discharge_disposition_id"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in source_data.columns
        ]

        if missing_columns:

            st.error(
                "Required dataset columns are missing: "
                + ", ".join(missing_columns)
            )

        else:

            st.subheader("Patient Information")

            st.info(
                "Enter the available patient information below. "
                "Other dataset features will use representative "
                "values derived from the uploaded dataset."
            )

            age_options = sorted(
                source_data["age"].dropna().astype(str).unique()
            )

            diagnosis_options = sorted(
                source_data["diag_1"].dropna().astype(str).unique()
            )

            discharge_options = sorted(
                source_data["discharge_disposition_id"]
                .dropna()
                .unique()
                .tolist(),
                key=lambda value: str(value)
            )

            if not age_options:
                age_options = ["[50-60)"]

            if not diagnosis_options:
                diagnosis_options = ["Unknown"]

            if not discharge_options:
                discharge_options = [1]

            patient_input = {}

            col1, col2 = st.columns(2)

            with col1:

                patient_input["age"] = st.selectbox(
                    "Patient Age Group",
                    age_options
                )

                patient_input["time_in_hospital"] = st.number_input(
                    "Time Spent in Hospital (Days)",
                    min_value=1,
                    max_value=30,
                    value=4
                )

                patient_input["num_lab_procedures"] = st.number_input(
                    "Number of Laboratory Procedures",
                    min_value=0,
                    max_value=100,
                    value=40
                )

                patient_input["num_medications"] = st.number_input(
                    "Number of Medications",
                    min_value=0,
                    max_value=100,
                    value=15
                )

                patient_input["number_outpatient"] = st.number_input(
                    "Number of Outpatient Visits",
                    min_value=0,
                    max_value=100,
                    value=0
                )

            with col2:

                patient_input["number_emergency"] = st.number_input(
                    "Number of Emergency Visits",
                    min_value=0,
                    max_value=100,
                    value=0
                )

                patient_input["number_inpatient"] = st.number_input(
                    "Number of Inpatient Visits",
                    min_value=0,
                    max_value=100,
                    value=0
                )

                patient_input["diag_1"] = st.selectbox(
                    "Primary Diagnosis",
                    diagnosis_options
                )

                patient_input["discharge_disposition_id"] = st.selectbox(
                    "Discharge Disposition",
                    discharge_options
                )

            st.markdown("---")

            st.subheader("Generate Prediction")

            if st.button(
                "Predict Readmission Risk",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Processing patient information and generating predictions..."
                    ):

                        patient_df = build_patient_input(
                            source_data,
                            patient_input
                        )

                        patient_scaled = prepare_patient_features(
                            patient_df,
                            summary
                        )

                        project_root = Path(__file__).resolve().parent

                        decision_tree_file = (
                            project_root
                            / "models"
                            / "decision_tree_model.pkl"
                        )

                        knn_file = (
                            project_root
                            / "models"
                            / "knn_model.pkl"
                        )

                        decision_tree_artifact = joblib.load(
                            decision_tree_file
                        )

                        knn_artifact = joblib.load(
                            knn_file
                        )

                        decision_tree_input = prepare_new_patient(
                            patient_scaled,
                            decision_tree_artifact["preprocessing"]
                        )

                        knn_input = prepare_new_patient(
                            patient_scaled,
                            knn_artifact["preprocessing"]
                        )

                        decision_tree_prediction = (
                            decision_tree_artifact["model"]
                            .predict(decision_tree_input)[0]
                        )

                        knn_prediction = (
                            knn_artifact["model"]
                            .predict(knn_input)[0]
                        )

                        prediction_results = {
                            "Decision Tree": int(
                                decision_tree_prediction
                            ),
                            "K-Nearest Neighbors": int(
                                knn_prediction
                            )
                        }

                        st.session_state.prediction_results = (
                            prediction_results
                        )

                        st.session_state.prediction_patient = (
                            patient_input.copy()
                        )

                    st.success(
                        "Patient prediction completed successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

            if st.session_state.prediction_results is not None:

                prediction_results = (
                    st.session_state.prediction_results
                )

                st.markdown("---")
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                for column, model_name in [
                    (col1, "Decision Tree"),
                    (col2, "K-Nearest Neighbors")
                ]:

                    prediction = prediction_results[model_name]

                    with column:

                        st.markdown(f"### {model_name}")

                        if prediction == 1:

                            st.error(
                                "Readmitted within 30 days"
                            )

                        else:

                            st.success(
                                "Not readmitted within 30 days"
                            )

                        st.metric(
                            "Predicted Class",
                            prediction
                        )

                st.markdown("---")
                st.subheader("Model Comparison")

                if (
                    prediction_results["Decision Tree"]
                    == prediction_results["K-Nearest Neighbors"]
                ):

                    st.info(
                        "Both models generated the same prediction."
                    )

                else:

                    st.warning(
                        "The two models generated different predictions."
                    )

                comparison_table = pd.DataFrame({
                    "Model": list(prediction_results.keys()),
                    "Predicted Class": list(
                        prediction_results.values()
                    ),
                    "Prediction": [
                        (
                            "Readmitted within 30 days"
                            if value == 1
                            else "Not readmitted within 30 days"
                        )
                        for value in prediction_results.values()
                    ]
                })

                st.dataframe(
                    comparison_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")
                st.subheader("Important Notice")

                st.warning(
                    "This system is an academic data mining project. "
                    "Its predictions are experimental and must not be "
                    "used as a medical diagnosis or as the basis for "
                    "actual patient treatment or discharge decisions. "
                    "The models currently have low recall for patients "
                    "readmitted within 30 days."
                )

st.markdown("---")

st.caption(
    "Hospital Readmission Risk Prediction System | Data Mining Project"
)
