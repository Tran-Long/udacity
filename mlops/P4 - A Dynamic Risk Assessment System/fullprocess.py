import logging
import os
from pathlib import Path
import pickle
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
from utils import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    cfg = Config(config_file=str(Path(__file__).parent / "config_prod.json"))
    # Check and read new data
    # first, read ingestedfiles.txt
    logger.info("Reading ingested data files...")
    ingested_file_path = os.path.join(
        cfg.output_folder_path, "ingestedfiles.txt")
    with open(ingested_file_path, "r") as f:
        ingested_files = f.readlines()
    ingested_files = [f.strip() for f in ingested_files]

    # second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    logger.info("Checking data drift...")
    prod_data_files = os.listdir(cfg.input_folder_path)
    new_data_exist = False
    for file in prod_data_files:
        if file not in ingested_files:
            new_data_exist = True
            break

    # Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process
    # here
    if new_data_exist:
        logger.info("Data drift detected. Ingesting new data")
        ingestion.merge_multiple_dataframe(
            cfg.input_folder_path,
            cfg.output_folder_path
        )
        new_data_path = os.path.join(cfg.output_folder_path, "finaldata.csv")
    else:
        logger.info("No data drift happened, exiting...")
        exit(0)

    # Checking for model drift
    # check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    logger.info("Validating current model with new data...")
    old_score = float(
        open(
            os.path.join(
                cfg.prod_deployment_path,
                "latestscore.txt"),
            "r").read())
    current_model_file_path = os.path.join(
        cfg.prod_deployment_path, "trainedmodel.pkl")
    new_score = scoring.score_model(
        current_model_file_path,
        new_data_path,
        store=False)
    if new_score >= old_score:
        logger.info("Model working fine with new data")
        exit(0)

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the
    # process here
    logger.info("Model drift detected")
    logger.info("Retraining model...")
    training.train_model(new_data_path, cfg.output_model_path)
    trained_model_path = os.path.join(
        cfg.output_model_path, "trainedmodel.pkl")
    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
    scoring.score_model(trained_model_path, test_data_path)

    # Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    logger.info("Re-deploy new model...")
    model_file_path = os.path.join(cfg.output_model_path, "trainedmodel.pkl")
    score_file_path = os.path.join(cfg.output_model_path, "latestscore.txt")
    ingest_file_path = os.path.join(
        cfg.output_folder_path,
        "ingestedfiles.txt")
    prod_deployment_path = cfg.prod_deployment_path
    deployment.store_model_into_pickle(
        [model_file_path, score_file_path, ingest_file_path],
        prod_deployment_path
    )
    # Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model
    logger.info("Diagnostics and reporting for the new re-deployed model")

    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
    reporting.score_model(test_data_path, model_file_path)

    diagnostics.model_predictions(model_file_path, test_data_path)
    diagnostics.dataframe_summary(test_data_path)
    diagnostics.missing_data_check(test_data_path)
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()

    logger.info("Finish")
