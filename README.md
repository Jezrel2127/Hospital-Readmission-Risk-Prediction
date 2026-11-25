
# Hospital Readmission Risk Prediction System

Data Mining Project on analysis of hospital patients data and experimental predictions of readmissions with the help of Decision Tree and K-Nearest Neighbors (KNN).

## Project Features

1. Dataset Loading
2. Data Quality Analysis
3. Noise Data Detection
4. Data Cleaning
5. Data Preprocessing
6. Data Visualization
7. Model Development
8. Model Evaluation
9. Readmission Risk Prediction

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Joblib
- Decision Tree
- K-Nearest Neighbors (KNN)

## Requirements

- Python 3.12
- Visual Studio Code
- Git

## Installation Guide

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Hospital_Readmission_Risk_Prediction
```

### 3. Create a Virtual Environment

```bash
py -3.12 -m venv .venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## Machine Learning Models

### Decision Tree
Uses Gini Impurity to split the dataset into decision branches.

### K-Nearest Neighbors
Uses the distance between patient records to classify new inputs.

## Evaluation Results

| Metric | Decision Tree | KNN |
|---|---:|---:|
| Accuracy | 88.71% | 87.71% |
| Precision | 0.00% | 7.69% |
| Recall | 0.00% | 0.89% |
| F1-Score | 0.00% | 1.60% |

KNN achieved the higher F1-Score among the two models. However, both models showed low recall for patients readmitted within 30 days.

## Important Notice

The project is intended for use solely as an educational tool in data mining. The predictions are experimental in nature and should never be used to diagnose any illness.
