import audioop
import codecs
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
import json
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


class DataCleaner:
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
    
    def resize_pad_trunc(self,df,max_ms=4000):
        # aud, max_ms

        for i in range(df.shape[0]):
            data = df.loc[i, 'Output']
            try:
                sig, framerate = librosa.load(data, sr=None, mono=False)
            except:
                logger.warning(
                    "Data is missing ("+str(data)+"), please check!")
                continue
            max_len = framerate // 1000 * max_ms
            trimmed=librosa.util.fix_length(sig, size=max_len)
            input = trimmed
            if(type(trimmed[0]) == list):
                input = trimmed[0]
            sf.write(data, input, framerate)
            logger.info("successfully resized audio")
    
    def shuffle_data(self, df, state):
        """
        df: meta_data dataframe that has the path info for the data
        """
        selection = df[df["Duration"] != 400]
        shuffled_meta = selection.sample(frac=1, random_state=state).reset_index().drop("index", axis=1)
        
        return shuffled_meta


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


    def make_stereo(self, df, output=False):
        for i in range(df.shape[0]):
            input_p = df.loc[i, "Feature"]
            if(output):
                input_p = df.loc[i, "Output"]
            
            output_p = df.loc[i, "Output"]
            try:
                ifile = wave.open(input_p)
                # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
            except:
                logger.warning(
                    "Data is missing ("+str(input_p)+"), please check!")
                continue
            (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
            assert comptype == 'NONE'  # Compressed not supported yet
            array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
            left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
            ifile.close()

            stereo = 2 * left_channel
            stereo[0::2] = stereo[1::2] = left_channel

            ofile = wave.open(output_p, 'w')
            ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
            ofile.writeframes(stereo)
            ofile.close()
            logger.info("successfully converted channel from mono to stereo")



    def standardize(self, df, output=False):
        # standardize to 44.1KHz
        for i in range(df.shape[0]):
            
            input_p = df.loc[i, 'Feature']
            if(output):
                input_p = df.loc[i, 'Output']
            output_p = df.loc[i, 'Output']
            try:
                ifile = wave.open(input_p)
            except:
                continue
            (nchannels, sampwidth, framerate, nframes,
             comptype, compname) = ifile.getparams()
            frames = ifile.readframes(nframes)
            ifile.close()
            ofile = wave.open(output_p, 'w')
            ofile.setparams(
                (nchannels, sampwidth, 44100, int(np.round(44100*nframes/framerate,0)), comptype, compname))
            converted = audioop.ratecv(frames, sampwidth, nchannels, framerate, 44100, None)
            ofile.writeframes(converted[0])
            ofile.close()
            logger.info("successfully standardized sample rate")
       
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
    
    # Recieving a file and creating a feature out of it

    def features_extractor(self,path):
        audio, _ = librosa.load(path, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        
        return mfccs_scaled_features

    # Now we iterate through every audio file and extract features 
    # Using Mel-Frequency Cepstral
    # This funtion will recieve a dataframe and return a data frame with features and target as a column
    
    def total_feature_extractor(self,meta_data):
        """
        meta_data: dataframe tha contains path to data and other info
        return type: a dataframe with a column called features that contains
        an array that 
        """
        extracted_features=[]
        for index_num,row in tqdm(meta_data.iterrows()):
            if (row['n_channel']!=400):
                file_name = row['Feature']
                final_class_labels=row['Target']
                data=self.features_extractor(file_name)
                extracted_features.append([data,final_class_labels])
            else:
                continue 

        extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','target'])

        logger.info("Successfully featurized!!!")
        
        return extracted_features_df   
    
    def meta_loader(self, path, type):
        """
        path: path of the files to be loaded
        type: type of the file to be loaded
        return: a dataframe of the loaded file
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
    
    
    
    def time_shift(self, df, shift, output=False):
        for i in range(df.shape[0]):
            input_p = df.loc[i, 'Feature']
            if(output):
                input_p = df.loc[i, 'Output']
            output_p = df.loc[i, 'Output']
            try:
                data, framerate = librosa.load(input_p, sr=None, mono=False)
            except:
                logger.warning(
                    "Data is missing ("+str(input_p)+"), please check!")
                continue
            mod_data = np.roll(data, int(shift))
            sf.write(output_p, mod_data, framerate)
        
    
          

