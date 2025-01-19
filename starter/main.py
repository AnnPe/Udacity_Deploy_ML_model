# Put the code for your API here.
from typing import Literal
import os
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference

folder_path = os.path.dirname(os.path.abspath(__file__))


# Declare the data object with its components and their type.
class InputItem(BaseModel):
    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?',
                       'Self-emp-inc', 'Without-pay', 'Never-worked']
    # fnlgt: int
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm',
                       'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th',
                       '1st-4th', 'Preschool', '12th']
    # education_num: int
    marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                            'Separated', 'Married-AF-spouse', 'Widowed']\
        = Field(alias="marital-status")

    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
                        'Other-service', 'Sales', 'Craft-repair', 'Transport-moving',
                        'Farming-fishing', 'Machine-op-inspct', 'Tech-support', '?',
                        '                      Protective-serv', 'Armed-Forces', 'Priv-house-serv']

    relationship: Literal['Not-in-family', 'Husband',
                          'Wife', 'Own-child', 'Unmarried', 'Other-relative']

    race: Literal['White',
                  'Black',
                  'Asian-Pac-Islander',
                  'Amer-Indian-Eskimo',
                  'Other']

    sex: Literal['Male', 'Female']

    # capital_gain: int= Field(alias="capital-gain")
    # capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")

    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico', 'South',
                            'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                            'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador',
                            'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic', 'El-Salvador',
                            'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia', 'Peru',
                            'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece',
                            'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']\
        = Field(alias="native-country")


app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
print(f"{folder_path}/model/")


def get_model():
    print(os.listdir())
    return joblib.load(f"{folder_path}/model/model.joblib"), joblib.load(
        f"{folder_path}/model/encoder.joblib"), joblib.load(f"{folder_path}/model/lb.joblib")


model, encoder, lb = get_model()


@app.get("/")
async def greetings():
    return {"message": "hello world"}


@app.post("/predict/")
async def predict_salary(item: InputItem):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    print(item)
    df = pd.DataFrame.from_dict([item.dict()], orient="columns")

    data = process_data(
        df,
        encoder=encoder,
        lb=lb,
        label=None,
        categorical_features=cat_features,
        training=False)[0]

    prediction = inference(model, data)[0]
    prediction = str(lb.inverse_transform(prediction)[0])
    return {"predictions": prediction}
