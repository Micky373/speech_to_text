# loading libraries
import librosa   #for audio processing
import librosa.display
import wave
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sys
import json
import logging
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")


# setting up logger
"""
initilize logger
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("../logs/clean_data.log")
formatter = logging.Formatter(
    "time: %(asctime)s, message: %(message)s \n")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# loading scripts
sys.path.insert(1, '../scripts')
sys.path.append("..")
sys.path.append(".")

from data_cleaning import DataCleaner
from data_viz import Data_Viz

DC = DataCleaner("../logs/preprocessing_notebook.log")
DV = Data_Viz()


# loading meta data
path = "../data/train"
output = "../data/train_new"

meta_data = DC.generate_metadata(path, output)
print(f"meta_data generated; shape: {meta_data.shape}")
logger.info("Meta_data created")


# adding duration column to meta data

DC.add_duration(meta_data)
selection = meta_data[meta_data["Duration"] != 400]
print(f"Duration column added:\n{selection.head(2)}")
logger.info("Adding duration column")

# standardizing sampling rate

print("\nStandardizing sampling rate...")
data_b =  wave.open('../data/train/wav/tr_10_tr01010.wav')  # checking before standardization
print(f"sampling rate before standardization: {data_b.getparams().framerate}") 

DC.standardize(meta_data)  # standardizing
data_a =  wave.open('../data/train_new/tr_10_tr01010.wav') # checking rate after standardizing 
print(f"sampling rate after standardization: {data_a.getparams().framerate}") 
logger.info("Sample rate standardized")

# Resizing Audio samples

print("\nResizing audio samples...")
meta_data= DC.add_duration(meta_data, output= True)
selection = meta_data[meta_data["Duration"] != 400]
print(f"Data before resizing: \n{selection['Duration'].head(2)}")

DC.resize_pad_trunc(meta_data, 10100) # resizing to 10 seconds

meta_data= DC.add_duration(meta_data, output= True)
selection = meta_data[meta_data["Duration"] != 400]
print(f"Data bafter resizing: \n{selection['Duration'].head(2)}")
logger.info("Audio resized")

# Convert mono to stereo

print("\nConverting to Stereo...")
print("adding new column for channel count")
meta_data= DC.channel_count(meta_data)
selection = meta_data[meta_data["Duration"] != 400]
print(f'number of channels before: \n{selection["n_channel"].value_counts()}')
DC.make_stereo(meta_data, True)
logger.info("Converted to stereo")

# Data Augumentation

print("Data Augumentation...")
DC.time_shift(meta_data, int(sample_rate/10), True)
print("Data Augumented")

# making sure everything is right
data =  wave.open('../data/train_new/tr_10_tr01010.wav')
print("the parameters are: ", data.getparams())
logger.info(f"{data.getparams()}")

# shuffling and splitting data

print("Splitting and saving data...")
dfs = DC.split(meta_data, 80, 2)
train = dfs[0]
test = dfs[1]
print(train.shape)
print(test.shape)

# saving data as json

DC.meta_saver(train, "../data/train_meta.csv", "csv")
DC.meta_saver(test, "../data/test_meta.csv", "csv")

logger.info("Metadata saved")