import pandas as pd
import numpy as np
# from regex import D
import sys
import wave
import struct
import soundfile as sf
import os
import librosa  # for audio processing
import librosa.display
import logging
import random
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

    def channel_count(self, df, output=False):
        """
        It identifies number of channels in the audio files
        and adds a new column with the identified number
        """
        n_list = []
        col = "Feature"
        if(output):
            col = "Output"
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

    def make_stereo(self, df):
        def everyOther(v, offset=0):
            return [v[i] for i in range(offset, len(v), 2)]
        for i in range(df.shape[0]):
            input_p = df.loc[i, "Feature"]
            output_p = df.loc[i, "Output"]
            try:
                ifile = wave.open(input_p)
                # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
            except:
                logger.warning(
                    "Data is missing ("+str(input_p)+"), please check!")
                continue
            (nchannels, sampwidth, framerate, nframes,
             comptype, compname) = ifile.getparams()
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
            ofile = wave.open(output_p, 'w')
            ofile.setparams(
                (2, sampwidth, framerate, nframes, comptype, compname))
            ofile.writeframes(left.tostring())
            ofile.writeframes(right.tostring())
            ofile.close()

            logger.info("successfully converted channel from mono to stereo")

    def standardize(self, df):
        # standardize to 44.1KHz
        for i in range(df.shape[0]):
            data = df.loc[i, 'Output']
            try:
                ifile = wave.open(data)
            except:
                continue
            (nchannels, sampwidth, framerate, nframes,
             comptype, compname) = ifile.getparams()
            ofile = wave.open(data, 'w')
            ofile.setparams(
                (nchannels, sampwidth, 44100, nframes, comptype, compname))
            ofile.close()
            logger.info("successfully standardized sample rate")
            
    def resize_pad_trunc(self,df,max_ms=4000):
        # aud, max_ms

        for i in range(df.shape[0]):
            data = df.loc[i, 'Output']
            
            # print(str(data).split("../")[-1])
            path=os.path.abspath(os.path.join(os.pardir, str(data).split("../")[-1]))
            
            sig, sr =sf.read(path)
            num_rows, sig_len = sig.shape
            max_len = sr // 1000 * max_ms
            if sig_len > max_len:
                # Truncate the signal to the given length
                # sig = sig[:, :max_len]
                trimmed=librosa.util.fix_length(sig, size=max_len)
                logger.info("successfully truncated audio")
                
            elif sig_len < max_len:
                # Length of padding to add at the beginning and end of the signal
                
                trimmed=librosa.util.fix_length(sig, size=max_len)
                logger.info("successfully padded audio")
                
            output='train_new'   
            audio_file = os.path.basename(data)
            directory = os.path.dirname(data)
            audio_file
            _,sampling_rate=librosa.load(data, sr=44100) 
            name = os.path.join(output, audio_file)
            sf.write(path, trimmed.T, sampling_rate)

            
            

