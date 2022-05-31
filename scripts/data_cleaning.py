import pandas as pd
import numpy as np
from regex import D
import sys
import wave
import struct

import logging
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
        formatter = logging.Formatter("time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    def channel_count(self, df, output=False):
        """
        It identifies number of channels in the audio files
        and adds a new column with the identified number
        """
        n_list=[]
        col = "Feature"
        if(output): 
            col = "Output"
        for i in range(df.shape[0]):
            try:
                data = wave.open(df.loc[i,col], mode = 'rb')
            except:
                n_list.append(400)  # 400 means the data is missing
                continue
            channel = data.getparams().nchannels
            n_list.append(channel)
        df["n_channel"] = n_list

        return df


    def generate_metadata(self, path, output):
        """
        extracts target and feature out of the trsTrain.txt file
        """
        meta_data =pd.read_csv(path+"/trsTrain.txt",sep="\t",header=None)
        meta_data.rename(columns = {0:'Target'}, inplace = True)
        meta_data['Feature'] = meta_data['Target'].apply(lambda x: x.split("</s>")[1].replace("(", "").replace(")", "").strip())
        meta_data['Target'] = meta_data['Target'].apply(lambda x: x.split("</s>")[0].replace("<s>", "").strip())
        meta_data['Feature'] = meta_data['Feature'].apply(lambda x: path+"/wav/"+x+".wav")
        meta_data['Output'] = meta_data['Feature'].apply(lambda x: x.replace(path+"/wav",output))
        
        return meta_data


    def make_stereo(self, file1, output):
        
        def everyOther (v, offset=0):
            return [v[i] for i in range(offset, len(v), 2)]
        ifile = wave.open(file1)
        # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
        frames = ifile.readframes(nframes * nchannels)
        ifile.close()
        out = struct.unpack_from("%dh" % nframes * nchannels, frames)
        # Convert 2 channels to numpy arrays
        if nchannels == 2:
            left = np.array(list(everyOther(out, 0)))
            right = np.array(list(everyOther(out, 1)))
        else:
            left = np.array(out)
            right = left
        ofile = wave.open(output, 'w')
        ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
        ofile.writeframes(left.tostring())
        ofile.writeframes(right.tostring())
        ofile.close()