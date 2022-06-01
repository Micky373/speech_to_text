import torch
import random

@staticmethod
def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if sig_len > max_len:
        # Truncate the signal to the given length
        sig = sig[:, :max_len]


