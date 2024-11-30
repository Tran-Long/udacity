import logging
import requests


logging.basicConfig(
    filename="./logs/infer_log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Request to the Heroku server
    url = "https://uda-mlops-prj3-9422cc18fa38.herokuapp.com/predict"

    response = requests.post(
        url=url,
        json={
            "age": 62,
            "workclass": "State-gov",
            "fnlgt": 202056,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Divorced",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    )
    logger.info(f"Status code: {response.status_code}")
    logger.info(response.json())
