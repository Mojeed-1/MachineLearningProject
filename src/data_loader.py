import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file from local path or mounted directory.
    """
    return pd.read_csv(path)
