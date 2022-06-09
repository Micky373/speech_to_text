# loading libraries
import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML

# loading parameters
with open("params.yaml", 'r') as fd:
    yaml = YAML()
    params = yaml.load(fd)

RATE = params['prepare']['RATE']
DURATION = params['prepare']['DURATION']

# setting up logger
"""
initilize logger
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/clean_data.log")
formatter = logging.Formatter(
    "time: %(asctime)s, message: %(message)s \n")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# loading scripts
sys.path.insert(1, './scripts')
sys.path.append("..")
sys.path.append(".")

from meta_cleaner import MetaCleaner
from audio_cleaner import AudioCleaner

MC = MetaCleaner("logs/preprocessing_notebook.log")
AC = AudioCleaner("logs/preprocessing_notebook.log")


# loading meta data
path = "data/train"
output = "data/train_new"

meta_data = MC.generate_metadata(path, output)
print(f"meta_data generated; shape: {meta_data.shape}")
logger.info("Meta_data created")

# building sklearn pipeline to process audio
pipe = AC.build_pipe([RATE, DURATION])
logger.info("Pipeline created")


# running batch data cleaning.
AC.batch_iterator(meta_data, pipe)    
logger.info("Batch cleaning completed")

# shuffling and splitting data
dfs = MC.split(meta_data, 80, 2)
train = dfs[0]
test = dfs[1]
print(train.shape)
print(test.shape)

# saving data as csv

MC.meta_saver(train, "../data/train_meta.csv", "csv")
MC.meta_saver(test, "../data/test_meta.csv", "csv")
logger.info("Splitted meta saved")