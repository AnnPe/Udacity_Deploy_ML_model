# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import os
import pandas as pd
from ml.data import process_data
import joblib
from ml.model import train_model, inference, compute_model_metrics
import json

folder_path = os.path.dirname(os.path.abspath(__file__))

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

def slice_inference(data, features, model, encoder, lb):
    r = {}
    for feature in features:
        for value in data[feature].unique():
            subset = data[data[feature] == value]
            X, y, _, __ = process_data(subset,categorical_features=features, training = False, encoder = encoder, lb = lb,label="salary")
            pred = model.predict(X)
            precision, recall, fscore = compute_model_metrics(y, pred)
            if feature in r.keys():
                r[feature][value] = {"precision" : precision, "recall" : recall, "fscore" : fscore}
            else:
                r[feature] = {}
    return r


if __name__ == "__main__":

    data = pd.read_csv(f"{folder_path}/data/census_cleaned.csv")    
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(test,categorical_features=cat_features, training = False, encoder = encoder, lb = lb,label="salary")

    model = train_model(X_train, y_train)
    predictions = inference(model , X_test)

    print (compute_model_metrics(y_test, predictions))

    joblib.dump(model, f"{folder_path}/model/model.joblib")
    joblib.dump(encoder, f"{folder_path}/model/encoder.joblib")
    joblib.dump(lb, f"{folder_path}/model/lb.joblib")

    r = slice_inference(test, cat_features,  model, encoder, lb)
    with open (f"{folder_path}/model/slice_inference.json", "w") as file:
        json.dump(r, file, indent = 4)

