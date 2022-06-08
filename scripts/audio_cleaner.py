import librosa
import numpy as np
import logging
import codecs

import json
import soundfile as sf
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Cleaner:

    def __init__(self, filehandler) -> None:
        """
        initilize logger
        """
        file_handler = logging.FileHandler(filehandler)
        formatter = logging.Formatter(
            "time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def change_rate(self, audio, sr):
        """
        audio: a list that contains audio sample and rate
        sr: new sampling rate to be applied
        return: resampled audio array with sampling rate
        """
        resampled = librosa.resample(audio[0], orig_sr=audio[1], target_sr=sr)
        
        logger.info("rate changed")
        
        return [resampled, sr]


    def change_to_stereo(self, audio):
        """
        audio: a list that contains audio sample and rate
        return: stereo audio array with sampling rate
        """
        y = np.asfortranarray(np.array([audio[0], audio[0]]))

        return [ y, audio[1]]


    def change_duration(self, audio, max_ms):
        """
        audio: a list that contains audio sample and rate
        max_ms: audio duration to be created
        return: resampled audio array with sampling rate
        """
        max_len = audio[1] // 1000 * max_ms
        trimmed=librosa.util.fix_length(audio[0], size=max_len)

        return [trimmed, audio[1]]


    def time_shift(self, audio):
        """
        audio: a list that contains audio sample and rate
        return: time shifted audio array with sampling rate
        """
        y_shift = audio[0]
        timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
        start = int(y_shift.shape[0] * timeshift_fac)
        if (start > 0):
            y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
        else:
            y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]

        return [y_shift, audio[1]]


    def load_audio(self, path):
        """
        path: path to audio file to be loadded
        return: array of audio samples and sampling rate.
        """
        samples, sample_rate = librosa.load(path, sr=None, mono=False)
        
        return [samples, sample_rate]


    def save_audio(self, path, audio):
        """
        audio: a list that contains audio sample and rate
        path: location and name of new file
        """
        if(len(audio[0])==2):
            with sf.SoundFile(path, 'w', audio[1], 2, 'PCM_24') as f:
                f.write(np.transpose(audio[0]))
        else:
            sf.write(path, audio[0], audio[1])


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
        
if __name__ == "__main__":
    inst = Cleaner()
    sam, rat = inst.load_audio("data/train/wav/tr_10_tr01010.wav")
    input = [sam, rat]
    output = inst.change_to_stereo(input)
    output2 = inst.change_duration(output, 8001)
    output3 = inst.change_rate(output2, 44100)
    output4 = inst.time_shift(output3)
    inst.save_audio('data/tr_10_tr01010.wav', output4)
    sam2, rat2 = inst.load_audio("data/tr_10_tr01010.wav")

    print(sam2.shape)