import pytest
from train_model import train_model, compute_model_metrics, inference, process_data
import sklearn
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

import os
import joblib


@pytest.fixture(scope="module")
def dataset():
    # Load a dataset (replace with your actual file path)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(f"{folder_path}/data/census_cleaned.csv")  .iloc[:100]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Select a subset of the data for testing
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    return X, y


@pytest.fixture(scope="module")
def model():
    folder_path = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(f"{folder_path}/model/model.joblib")


def test_train_model(dataset):
    """Test that train_model returns a fitted RandomForestClassifier."""

    X, y = dataset

    model = train_model(X, y)
    assert isinstance(
        model, sklearn.ensemble._forest.RandomForestClassifier), "train_model did not return a RandomForestClassifier."
    try:
        check_is_fitted(model)
    except BaseException:
        pytest.fail("Model is not fitted.")


def test_compute_model_metrics(dataset):
    _, y = dataset
    """Test that compute_model_metrics returns precision, recall, and fbeta as floats."""
    precision, recall, fbeta = compute_model_metrics(y, [1, ] * len(y))

    assert isinstance(precision, float), "Precision is not a float."
    assert isinstance(recall, float), "Recall is not a float."
    assert isinstance(fbeta, float), "Fbeta is not a float."


def test_inference(dataset):
    """Test that inference returns a numpy array."""
    X, y = dataset

    m = train_model(X, y)
    preds = inference(m, X)
    assert isinstance(
        preds, np.ndarray), "Inference did not return a numpy array."
