import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analyze.utils.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from analyze.utils.read_file import read_file, sample_data

USE_SAMPLING = True
SAMPLE_SIZE = 10000


def preprocess_data(file_path, selected_features):
    """
    Preprocess the dataset: read, clean, and normalize the data.
    Args:
        file_path (str): Path to the dataset file.
        selected_features (list): List of feature names to select.
    Returns:
        pd.DataFrame: Normalized dataset with selected features.
    """
    # Read and clean the dataset
    df = read_file(file_path, selected_features)
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Separate features and labels
    features = df.drop(columns=["Label"])
    labels = df["Label"]

    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Combine normalized features with labels
    df_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    df_normalized["Label"] = labels.values

    return df_normalized


def evaluate_classifier(name, classifier, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a classifier with properly formatted output.
    Args:
        name (str): Name of the classifier.
        classifier (object): Classifier instance.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """
    print(f"\n{'=' * 40}")
    print(f"{'Training and Evaluating Classifier:':<30} {name}")
    print(f"{'=' * 40}")

    # Measure training time
    start_train = time.perf_counter()
    classifier.fit(X_train, y_train)
    end_train = time.perf_counter()

    # Measure prediction time
    start_predict = time.perf_counter()
    y_pred = classifier.predict(X_test)
    end_predict = time.perf_counter()

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="BENIGN", average="binary")
    recall = recall_score(y_test, y_pred, pos_label="BENIGN", average="binary")
    cm = confusion_matrix(y_test, y_pred)
    false_alarm_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])

    # Display results
    print(f"{'Metric':<25}{'Value':>15}")
    print(f"{'-' * 40}")
    print(f"{'Accuracy:':<25}{accuracy:.4f}")
    print(f"{'Precision:':<25}{precision:.4f}")
    print(f"{'Detection Rate (Recall):':<25}{recall:.4f}")
    print(f"{'False Alarm Rate:':<25}{false_alarm_rate:.4f}")

    # Pretty confusion matrix
    print(f"\n{'Confusion Matrix:':<25}")
    print(f"{'-' * 40}")
    print(f"{'':<20}{'Predicted BENIGN':<20}{'Predicted MALICIOUS'}")
    print(f"{'Actual BENIGN':<20}{cm[0, 0]:<20}{cm[0, 1]}")
    print(f"{'Actual MALICIOUS':<20}{cm[1, 0]:<20}{cm[1, 1]}")

    # Display runtime
    print(f"\n{'Runtime':<25}{'Time (seconds)':>15}")
    print(f"{'-' * 40}")
    print(f"{'Training Time:':<25}{end_train - start_train:.4f}")
    print(f"{'Prediction Time:':<25}{end_predict - start_predict:.4f}")
    print(f"{'=' * 40}\n")

    # Plot evaluation metrics and results
    plot_confusion_matrix(y_test, y_pred, file_name=f"confusion_matrix_{name}.png")
    plot_roc_curve(classifier, X_test, y_test, file_name=f"roc_curve_{name}.png")
    plot_precision_recall_curve(
        classifier, X_test, y_test, file_name=f"precision_recall_curve_{name}.png"
    )


def main():
    selected_features = [
        "Bwd Packet Length Mean",
        "Average Packet Size",
        "Active Max",
        "Idle Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Bwd Packet Length Std",
        "Bwd Packet Length Max",
        "Label",
    ]

    # Preprocess data
    file_path = "./data/ddos.csv"
    df_normalized = preprocess_data(file_path, selected_features)

    if USE_SAMPLING:
        # Sample data for development/testing
        df_normalized = sample_data(df_normalized, sample_size=SAMPLE_SIZE)

    # Split data into features and labels
    X = df_normalized.drop(columns=["Label"])
    y = df_normalized["Label"]

    # Plot class distribution and feature correlation
    plot_class_distribution(y, file_name="class_distribution.png")
    plot_feature_correlation(df_normalized, file_name="feature_correlation.png")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define classifiers
    classifiers = {
        "SVC": SVC(random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        evaluate_classifier(name, clf, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
