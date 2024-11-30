import logging
from pathlib import Path
import os
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file=str(Path(__file__).parent / 'config.json')):
        logger.info("Loading config file: %s" % os.path.basename(config_file))
        self.data = json.load(open(config_file, "r"))
        for key in self.data:
            setattr(self, key, os.path.join(os.getcwd(), self.data[key]))


def preprocess_data(
        data,
        label="exited",
        remove_features=["corporation"]
):
    y = data[label]
    X = data.drop(*[remove_features + [label]], axis=1)
    return X, y
