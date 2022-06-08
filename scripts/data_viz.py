import matplotlib.pyplot as plt
import pandas as pd
import wave
import numpy as np

<<<<<<< HEAD
=======

>>>>>>> 32b92ffc307b4c05dd623b4a59f4d684733f2631
class Data_Viz():
    """
    any data visualization
    methods are found here (tables and charts)
    """
<<<<<<< HEAD
=======

>>>>>>> 32b92ffc307b4c05dd623b4a59f4d684733f2631
    def __init__(self) -> None:
        pass

    def visualize(self, path: str):
<<<<<<< HEAD
    
        # reading the audio file
        raw = wave.open(path)
        signal = raw.readframes(-1)
        signal = np.frombuffer(signal, dtype ="int16")
        
        # gets the frame rate
        f_rate = raw.getframerate()
        time = np.linspace(
            0, # start
            len(signal) / f_rate,
            num = len(signal)
        )
        plt.figure(figsize=(15,5))
=======

        # reading the audio file
        raw = wave.open(path)
        signal = raw.readframes(-1)
        signal = np.frombuffer(signal, dtype="int16")

        # gets the frame rate
        f_rate = raw.getframerate()
        time = np.linspace(
            0,  # start
            len(signal) / f_rate,
            num=len(signal)
        )
        plt.figure(figsize=(15, 5))
>>>>>>> 32b92ffc307b4c05dd623b4a59f4d684733f2631
        plt.title("Sound Wave")
        plt.xlabel("Time")
        plt.plot(time, signal)
        plt.show()

<<<<<<< HEAD

=======
>>>>>>> 32b92ffc307b4c05dd623b4a59f4d684733f2631
    def plot_spec(self, data: np.array, sr: int) -> None:
        '''
        Function for plotting spectrogram along with amplitude wave graph
        '''
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].title.set_text(f'Shfiting the wave by Times {sr/10}')
        ax[0].specgram(data, Fs=2)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0, 1, len(data)), data)
<<<<<<< HEAD
=======
        plt.show()
>>>>>>> 32b92ffc307b4c05dd623b4a59f4d684733f2631
