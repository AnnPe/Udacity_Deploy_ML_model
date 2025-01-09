# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import os
import pandas as pd
from ml.data import process_data
import joblib
from ml.model import train_model, inference, compute_model_metrics

folder_path = os.path.dirname(os.path.abspath(__file__))

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(f"{folder_path}/data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

print (train.columns)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(test,categorical_features=cat_features, training = False, encoder = encoder, lb = lb,label="salary")

print ("X_test.shape", X_test.shape)
model = train_model(X_train, y_train)
predictions = inference(model , X_test)
print(compute_model_metrics(y_test,predictions))

print (predictions)
test["pred"] = predictions

print (test[test.pred == 0].iloc[0])

print (test[test.pred == 1].iloc[0])

print (f"{folder_path}/model/")
joblib.dump(model, f"{folder_path}/model/model.joblib")
joblib.dump(encoder, f"{folder_path}/model/encoder.joblib")
joblib.dump(lb, f"{folder_path}/model/lb.joblib")

# Proces the test data with the process_data function.

# Train and save a model.
