import logging
import os
from pathlib import Path
import requests
from utils import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    URL = "http://127.0.0.1:8000/"
    # cfg = Config()
    cfg = Config(config_file=str(Path(__file__).parent / "config_prod.json"))
    api_returns_file_path = os.path.join(
        cfg.output_model_path, "apireturns.txt")
    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
    logger.info("[APICALL] Making request to /prediction")
    response_prediction = requests.post(
        URL + 'prediction',
        json={
            "input_file_path": test_data_path}).content.decode()
    logger.info("[APICALL] Making request to /scoring")
    response_scoring = requests.get(URL + "scoring").content.decode()
    logger.info("[APICALL] Making request to /summarystats")
    response_stat = requests.get(URL + "summarystats").content.decode()
    logger.info("[APICALL] Making request to /diagnostics")
    response_diagnostics = requests.get(URL + "diagnostics").content.decode()

    responses = [
        response_prediction + "\n",
        response_scoring + "\n",
        response_stat + "\n",
        response_diagnostics + "\n"
    ]

    # write the responses to your workspace
    logger.info(f"[APICALL] Writing all results to {api_returns_file_path}")
    with open(api_returns_file_path, "w") as f:
        f.writelines(responses)
