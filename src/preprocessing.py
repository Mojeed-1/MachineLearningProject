import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    df: pd.DataFrame,
    label_column: str = "label",
    scale: bool = True,
):
    """
    Preprocess input dataframe for training or inference.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data
    label_column : str or None
        Name of label column. If None, inference mode.
    scale : bool
        Whether to scale numeric features

    Returns
    -------
    X : pd.DataFrame
        Processed features
    y : pd.Series or None
        Labels (None during inference)
    """

    df = df.copy()

    if label_column and label_column in df.columns:
        y = df[label_column]
        X = df.drop(columns=[label_column])
    else:
        X = df
        y = None
    X = X.fillna(0)

    if scale:
        scaler = StandardScaler()
        X[:] = scaler.fit_transform(X)

    return X, y
