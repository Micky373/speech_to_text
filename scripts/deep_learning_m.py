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

  def spectrogram_from_sample(samples, fft_length=256, sample_rate=2, hop_length=128):
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

  def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        
    return np.transpose(np.log(pxx[:ind, :] + eps))

class Mapping():

  def text_to_int_sequence(text,char_map):
    """ Convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            # print("checking character " + c + " in map:")
            # print(char_map)
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence

  def int_sequence_to_text(int_sequence,index_map):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        ch = index_map[c]
        text.append(ch)
    return text

class WordError():

  # Code adapted from https://martin-thoma.com/word-error-rate-calculation/
  def wer(r, h):
      """
      Calculation of WER with Levenshtein distance.

      Works only for iterables up to 254 elements (uint8).
      O(nm) time ans space complexity.

      Parameters
      ----------
      r : list
      h : list

      Returns
      -------
      int

      Examples
      --------
      >>> wer("who is there".split(), "is there".split())
      1
      >>> wer("who is there".split(), "".split())
      3
      >>> wer("".split(), "who is there".split())
      3
      """
      # initialisation
      import numpy
      d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
      d = d.reshape((len(r)+1, len(h)+1))
      for i in range(len(r)+1):
          for j in range(len(h)+1):
              if i == 0:
                  d[0][j] = j
              elif j == 0:
                  d[i][0] = i

      # computation
      for i in range(1, len(r)+1):
          for j in range(1, len(h)+1):
              if r[i-1] == h[j-1]:
                  d[i][j] = d[i-1][j-1]
              else:
                  substitution = d[i-1][j-1] + 1
                  insertion    = d[i][j-1] + 1
                  deletion     = d[i-1][j] + 1
                  d[i][j] = min(substitution, insertion, deletion)

      return d[len(r)][len(h)]