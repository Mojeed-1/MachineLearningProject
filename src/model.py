from sklearn.linear_model import LogisticRegression


def get_model(C=1.0, max_iter=100):
    return LogisticRegression(C=C, max_iter=max_iter)
