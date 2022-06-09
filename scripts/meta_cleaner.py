import pandas as pd
import numpy as np
# from regex import D
import sys
import wave
import codecs
from tqdm import tqdm
import array
import json
import audioop
import soundfile as sf
import librosa  # for audio processing
import librosa.display
import logging
import soundfile as sf
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# importing scripts
sys.path.insert(1, '..')
sys.path.append("..")
sys.path.append(".")


class MetaCleaner:
    """
    class that handles data cleaning.
    """

    def __init__(self, filehandler) -> None:
        """
        initilize logger
        """
        file_handler = logging.FileHandler(filehandler)
        formatter = logging.Formatter(
            "time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    def get_max_dur(self, df):
        """
        df: meta_data dataframe
        return: maximum duration.
        """
        df = df.loc[df["Duration"]!=400]
        df["Duration"] = df["Duration"].astype(int)
        df_sorted = df.sort_values(by="Duration", ascending=False).reset_index()
        max_dur = (int(df_sorted.head()["Duration"][0])+1)*1000
        print("maximum duration: "+str(max_dur/1000))

        return max_dur




    def meta_loader(self, path, type):
        """
        path: path to files to be loaded
        type: type of the file to be loaded
        return: a dataframe version of the loaded file
        """
        if (type == "json"):
            fileObject = open(path, "r")
            jsonContent = fileObject.read()
            aList = json.loads(jsonContent)
            df = pd.DataFrame.from_dict(eval(aList))
            df["Feature"] = df["Feature"].apply(lambda x: x.replace("\\", ""))
            df["Output"] = df["Output"].apply(lambda x: x.replace("\\", ""))
        elif(type=="csv"):
            df = pd.read_csv(path)
        else:
            print("Only json and csv files are loaded")
            logger.warning("Format Unknown")

        logger.info("Dataframe successfully loaded")

        return df



    # saving data 
    def meta_saver(self, df, path, type):
        """
        df: dataframe to save 
        path: location and name of file
        type: saving type: csv or json
        """
        if(type == "json"):
            file_json = df.to_json(orient="columns")
            with codecs.open(path, 'w', encoding='utf-8') as f:
                json.dump(file_json, f, ensure_ascii=False)
        elif(type == "csv"):
            df.to_csv(path)
        else:
            print("Only csv and json file formats are allowed!")
            logger.warning("format Unknown")

        logger.info("Dataframe successfully saved as "+type+" file")



    # splitting data 
    def split(self, df, tr, state):
        """
        df: meta data to be splitted
        tr: percentage of train data set
        state: the state of sampling for repeating split
        """
        shuffled = self.shuffle_data(df, state)
        train_index = round(len(shuffled)*(tr/100))
        train_df = shuffled.head(train_index)
        test_df = shuffled.loc[train_index:len(shuffled), :]

        return [train_df, test_df]



    # dataset shuffling
    def shuffle_data(self, df, state):
        """
        df: meta_data dataframe that has the path info for the data
        """
        selection = df[df["Duration"] != 400]
        shuffled_meta = selection.sample(frac=1, random_state=state).reset_index().drop("index", axis=1)
        
        return shuffled_meta


    def channel_count(self, df, output=False):
        """
        It identifies number of channels in the audio files
        and adds a new column with the identified number
        """
        n_list = []
        if(output):
            col = "Output"
        else:
            col = "Feature"
        for i in range(df.shape[0]):
            try:
                data = wave.open(df.loc[i, col], mode='rb')
            except:
                n_list.append(400)  # 400 means the data is missing
                continue
            channel = data.getparams().nchannels
            n_list.append(channel)
        df["n_channel"] = n_list

        logger.info("new column successfully added: channels count")

        return df


    def add_duration(self, df, output=False):
        d_list = []
        if(output):
            col = "Output"
        else:
            col = "Feature"
        for i in range(df.shape[0]):
            try:
                data = wave.open(df.loc[i, col], mode='rb')
            except:
                d_list.append(400)  # 400 means the data is missing
                continue
            frames = data.getnframes()
            rate = data.getframerate()
            duration = frames / float(rate)
            d_list.append(duration)
        df["Duration"] = d_list

        logger.info("new column successfully added: Duration")

        return df


    def generate_metadata(self, path, output):
        """
        extracts target and feature out of the trsTrain.txt file
        """
        meta_data = pd.read_csv(path+"/trsTrain.txt", sep="\t", header=None)
        meta_data.rename(columns={0: 'Target'}, inplace=True)
        meta_data['Feature'] = meta_data['Target'].apply(
            lambda x: x.split("</s>")[1].replace("(", "").replace(")", "").strip())
        meta_data['Target'] = meta_data['Target'].apply(
            lambda x: x.split("</s>")[0].replace("<s>", "").strip())
        meta_data['Feature'] = meta_data['Feature'].apply(
            lambda x: path+"/wav/"+x+".wav")
        meta_data['Output'] = meta_data['Feature'].apply(
            lambda x: x.replace(path+"/wav", output))
        

        logger.info("meta data successfully generated")

        return meta_data

