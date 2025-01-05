import os

import pandas as pd


def read_file(file_path: str, selected_features: list[str]) -> pd.DataFrame | pd.Series:
    """
    Reads a CSV file, trims leading and trailing spaces from column names,
    and selects the specified features.

    Arguments:
        file_path: str - The path to the CSV file.
        selected_features: List[str] - List of feature names to select.

    Returns:
        pd.DataFrame - The cleaned DataFrame with selected features.
    """
    # Get the absolute path of the CSV file
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )  # Go to the project root directory
    full_file_path = os.path.join(
        project_root, file_path
    )  # Combine with the relative file path

    # Load the dataset
    df = pd.read_csv(full_file_path)

    # Trim leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Select only the relevant columns (features)
    df_selected = df[selected_features]

    return df_selected


def sample_data(
    df: pd.DataFrame, sample_size: int = 10000, random_state: int = 42
) -> pd.DataFrame:
    """
    Sample the dataset for development purposes.

    Arguments:
        df: Original DataFrame.
        sample_size: Number of rows to sample.
        random_state: Seed for reproducibility.

    Returns:
        Sampled DataFrame.
    """
    return df.sample(n=sample_size, random_state=random_state)
