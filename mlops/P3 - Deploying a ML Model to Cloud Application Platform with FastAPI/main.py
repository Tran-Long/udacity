# Put the code for your API here.
from typing import Dict
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
# import hydra
import yaml

from ml import process_data, inference


app = FastAPI()


class CensusInputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 38,
                    "workclass": "Private",
                    "fnlgt": 8071,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "Black",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }
    }


@app.get(path="/")
def welcome_root():
    return {"message": "Welcome to the Project 3 of the Udacity MLOps course!"}


@app.get(path="/new")
def new():
    return {"message": "Testing CI/CD successfully!"}


@app.post(path="/predict")
# @hydra.main(config_path=".", config_name="config", version_base="1.2")
async def prediction(input_data: CensusInputData) -> Dict[str, str]:
    """
    API for get prediction from the model with data from POST request.
    Args:
        input_data (BasicInputData) : Instance of a BasicInputData object.
    Returns:
        dict: Dictionary containing the model output.
    """
    with open("./config.yaml", "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    [encoder, lb, model] = pickle.load(
        open(config["model"]["saved_model_path"], "rb"))
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.model_dump(by_alias=True).items()}, index=[0]
    )

    processed_input_data, _, _, _ = process_data(
        X=input_df,
        categorical_features=config['data']['cat_features'],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, processed_input_data)
    return {"Result": ">50K" if int(prediction[0]) == 1 else "<=50K"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=5000)
