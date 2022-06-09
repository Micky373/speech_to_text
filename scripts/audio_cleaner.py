import librosa
import numpy as np
import logging
import codecs
import os
import math
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import json
import soundfile as sf
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AudioCleaner:

    def __init__(self, filehandler) -> None:
        """
        initilize logger
        """
        file_handler = logging.FileHandler(filehandler)
        formatter = logging.Formatter(
            "time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    def batch_iterator(self, meta, pipe):
        """
        meta: dataframe with audio path information
        pipe: data cleaning pipe to take each loaded audio through
        return: none
        """
    
        for i in range(len(meta)):

            if(i==1):
                print("Audio Processing Started...")

            try:
                sam, rat = self.load_audio(meta.loc[i, "Feature"])
                audio = [sam, rat]
            except:
                continue
            
            p_audio = pipe.fit_transform(audio)
            self.save_audio(p_audio, meta.loc[i, "Output"])
            # progress update

            


    def build_pipe(self, param):
        """
        ACL: AudioCleaner class instance
        Param: fixed parameters to apply to samples
        """
        pipe = Pipeline(steps = [
                                ("make stereo", FunctionTransformer(self.change_to_stereo)),
                                ("change rate", FunctionTransformer(self.change_rate, kw_args={"sr":param[0]})), 
                                ("change length", FunctionTransformer(self.change_duration, kw_args={"max_ms":param[1]})),
                                ("augument", FunctionTransformer(self.time_shift))
                                ])
        print("Pipeline Ready!\n")
        return pipe



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


    def save_audio(self, audio, path):
        """
        audio: a list that contains audio sample and rate
        path: location and name of new file
        """
        
        if(len(audio[0])==2):
            with sf.SoundFile(path, 'w', audio[1], 2, 'PCM_24') as f:
                f.write(np.transpose(audio[0]))
        else:
            sf.write(path, audio[0], audio[1])

        
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