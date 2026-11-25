
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def prepare_data(X, y):

    X = X.copy()
    y = pd.Series(y, index=X.index)

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int).copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    rng = np.random.default_rng(42)

    train_indices = []
    test_indices = []

    for class_value in np.unique(y):

        class_indices = np.where(
            y.to_numpy() == class_value
        )[0]

        rng.shuffle(class_indices)

        split_point = int(len(class_indices) * 0.80)

        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices].to_numpy()
    y_test = y.iloc[test_indices].to_numpy()

    medians = X_train.median().fillna(0)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    minimum = X_train.min()
    maximum = X_train.max()

    denominator = (maximum - minimum).replace(0, 1)

    X_train = (X_train - minimum) / denominator
    X_test = (X_test - minimum) / denominator

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    variances = X_train.var()

    selected_features = (
        variances.sort_values(ascending=False)
        .head(min(30, len(variances)))
        .index.tolist()
    )

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    preprocessing = {
        "columns": X.columns.tolist(),
        "medians": medians,
        "minimum": minimum,
        "maximum": maximum,
        "selected_features": selected_features
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing
    )
def prepare_new_patient(patient_data, preprocessing):

    X = patient_data.copy()
    X = X.reindex(
        columns=preprocessing["columns"],
        fill_value=0
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    X = X.fillna(preprocessing["medians"])
    
    minimum = preprocessing["minimum"]
    maximum = preprocessing["maximum"]

    denominator = (maximum - minimum).replace(0, 1)

    X = (X - minimum) / denominator
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = X.reindex(
        columns=preprocessing["selected_features"],
        fill_value=0
    )

    return X

class SimpleDecisionTree:

    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):

        if len(y) == 0:
            return 0

        probabilities = np.bincount(
            y,
            minlength=2
        ) / len(y)

        return 1 - np.sum(probabilities ** 2)

    def build_tree(self, X, y, depth=0):

        positive = int(np.sum(y == 1))
        negative = int(np.sum(y == 0))

        prediction = 1 if positive > negative else 0

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"prediction": prediction}

        best_gain = 0
        best_split = None
        parent_gini = self.gini(y)

        for feature in range(X.shape[1]):

            values = X[:, feature]

            thresholds = np.unique(
                np.quantile(
                    values,
                    np.linspace(0.1, 0.9, 9)
                )
            )

            for threshold in thresholds:

                left_mask = values <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() == 0
                    or right_mask.sum() == 0
                ):
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                weighted_gini = (
                    len(left_y) / len(y) * self.gini(left_y)
                    + len(right_y) / len(y) * self.gini(right_y)
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)

        if best_split is None:
            return {"prediction": prediction}

        feature, threshold = best_split

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(
                X[left_mask],
                y[left_mask],
                depth + 1
            ),
            "right": self.build_tree(
                X[right_mask],
                y[right_mask],
                depth + 1
            ),
            "prediction": prediction
        }

    def fit(self, X, y):

        self.tree = self.build_tree(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=int)
        )

        return self

    def predict_one(self, row, node):

        if "feature" not in node:
            return node["prediction"]

        if row[node["feature"]] <= node["threshold"]:
            return self.predict_one(row, node["left"])

        return self.predict_one(row, node["right"])

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        return np.array([
            self.predict_one(row, self.tree)
            for row in X
        ])
    
class SimpleKNN:

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)
        predictions = []

        for start in range(0, len(X), 100):

            batch = X[start:start + 100]

            distances = (
                np.sum(batch ** 2, axis=1)[:, None]
                + np.sum(self.X_train ** 2, axis=1)[None, :]
                - 2 * batch @ self.X_train.T
            )

            distances = np.maximum(distances, 0)

            nearest_indices = np.argpartition(
                distances,
                self.n_neighbors - 1,
                axis=1
            )[:, :self.n_neighbors]

            for indices in nearest_indices:

                votes = self.y_train[indices]

                prediction = np.bincount(
                    votes,
                    minlength=2
                ).argmax()

                predictions.append(prediction)

        return np.array(predictions)

def evaluate_model(y_true, predictions):

    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    true_negative = int(
        np.sum((y_true == 0) & (predictions == 0))
    )

    false_positive = int(
        np.sum((y_true == 0) & (predictions == 1))
    )

    false_negative = int(
        np.sum((y_true == 1) & (predictions == 0))
    )

    true_positive = int(
        np.sum((y_true == 1) & (predictions == 1))
    )

    accuracy = (
        (true_positive + true_negative) / len(y_true)
        if len(y_true) else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    confusion = np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])

    report = (
        "Class 0 (Not readmitted)\n"
        f"Precision: {true_negative / (true_negative + false_negative) if true_negative + false_negative else 0:.4f}\n"
        f"Recall: {true_negative / (true_negative + false_positive) if true_negative + false_positive else 0:.4f}\n\n"
        "Class 1 (Readmitted within 30 days)\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-Score: {f1_score:.4f}\n"
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Confusion Matrix": confusion,
        "Classification Report": report
    }

def train_models(X, y):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing
    ) = prepare_data(X, y)

    models = {
        "Decision Tree": SimpleDecisionTree(
            max_depth=5,
            min_samples_split=20
        ),
        "K-Nearest Neighbors": SimpleKNN(
            n_neighbors=5
        )
    }

    results = {}
    trained_models = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[model_name] = evaluate_model(
            y_test,
            predictions
        )

        trained_models[model_name] = model

    project_root = Path(__file__).resolve().parent.parent
    models_folder = project_root / "models"
    models_folder.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": trained_models["Decision Tree"],
            "preprocessing": preprocessing
        },
        models_folder / "decision_tree_model.pkl"
    )

    joblib.dump(
        {
            "model": trained_models["K-Nearest Neighbors"],
            "preprocessing": preprocessing
        },
        models_folder / "knn_model.pkl"
    )

    return results, trained_models, X_test, y_test
