import os
import numpy as np
from numpy.lib.stride_tricks import as_strided

class Separating():

  def train_valid(src,dest_train,dest_valid):
    i = 0
    # fetch all files
    for file_name in os.listdir(src):
        # construct full file path  
        if i < ((len(os.listdir(src)))*0.8):
          source_train = src + file_name
          destination_train = dest_train + file_name
          # copy only files
          if os.path.isfile(source_train):
            shutil.copy(source_train, destination_train)
        else:
          source_valid = src + file_name
          destination_valid = dest_valid + file_name
            # copy only files
          if os.path.isfile(source_valid):
                shutil.copy(source_valid, destination_valid)
        i+=1

class Replacing():

  # replace redundant letters
  def replacer(text):
      replace_list = """ሐ ሑ ሒ ሓ ሔ ሕ ሖ ጸ ጹ ጺ ጻ ጼ ጽ ጾ ኰ ኲ ጿ ኸ""".split(" ")
      ph = """ሀ ሁ ሂ ሀ ሄ ህ ሆ ፀ ፁ ፂ ፃ ፄ ፅ ፆ ኮ ኳ ፇ ኧ""".split(" ")
      for l in range(len(replace_list)):
        text = text.replace(replace_list[l], ph[l])
      return text

class Spectogram():

  def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs