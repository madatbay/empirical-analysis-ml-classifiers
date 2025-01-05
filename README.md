# Machine Learning Classifiers Time Complexity Analysis

## Overview

This project focuses on the time complexity analysis of machine learning classifiers, specifically analyzing the empirical performance of three classifiers: Support Vector Classifier (SVC), Random Forest, and K-Nearest Neighbors (KNN). The objective is to measure the time complexity of these algorithms using empirical analysis and compare their computational efficiency, accuracy, precision, recall, and false alarm rates. The study investigates the time required for both training and prediction, comparing them against theoretical time complexity to determine the most efficient classifier for large-scale applications.

## Requirements

- `uv` Python environment management tool
- Python 3.13

## Installation

To set up the project, ensure you have Python 3.13 installed with `uv`. After running `uv` scripts required dependencies will automatically installed.

Alternatively, you can install the project dependencies using pyproject.toml if you are familiar with uv:

```bash
uv sync
```

## Running the Project

To run the entire project, use the following command:

```bash
uv run analyze
```

This will execute the complete workflow, including:

1. Data preprocessing
2. Training and evaluating the classifiers (SVC, Random Forest, KNN)
3. Generating performance metrics (accuracy, precision, recall, etc.)
4. Saving output visualizations such as confusion matrices, ROC curves, and precision-recall curves
5. Outputting results to CSV files
