import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# Ensure the assets folder exists
os.makedirs("assets", exist_ok=True)


def plot_class_distribution(y, file_name="class_distribution.png"):
    """
    Saves a bar plot showing the distribution of classes in the dataset.

    Args:
        y (pd.Series): Target labels.
        file_name (str): Name of the output file.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.savefig(f"assets/{file_name}")
    plt.close()


def plot_feature_correlation(df, file_name="feature_correlation.png"):
    """
    Plot correlation matrix for the features and save it as a file.
    Args:
        df (pd.DataFrame): The dataframe with features.
        file_name (str): The name of the file to save the plot.
    """
    # Select only numerical columns for correlation
    df_numerical = df.select_dtypes(include=["float64", "int64"])

    # Compute the correlation matrix
    correlation_matrix = df_numerical.corr()

    # Create the heatmap plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"assets/{file_name}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, file_name="confusion_matrix.png"):
    """
    Saves a heatmap of the confusion matrix.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        file_name (str): Name of the output file.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN", "MALICIOUS"],
        yticklabels=["BENIGN", "MALICIOUS"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"assets/{file_name}")
    plt.close()


def plot_roc_curve(classifier, X_test, y_test, file_name="roc_curve.png"):
    """
    Plot the ROC curve for the classifier and save it as a file.
    Args:
        classifier (object): The trained classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test data.
        file_name (str): The name of the file to save the plot.
    """
    # Get predicted probabilities
    y_prob = classifier.predict_proba(X_test)[
        :, 1
    ]  # Probabilities for the positive class

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(
        y_test, y_prob, pos_label="BENIGN"
    )  # Adjust pos_label if needed
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"assets/{file_name}")
    plt.close()


def plot_precision_recall_curve(
    classifier, X_test, y_test, file_name="precision_recall_curve.png"
):
    """
    Saves a precision-recall curve for the classifier.

    Args:
        classifier (object): Trained classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.
        file_name (str): Name of the output file.
    """
    y_prob = classifier.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label="BENIGN")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve", color="purple")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.savefig(f"assets/{file_name}")
    plt.close()
