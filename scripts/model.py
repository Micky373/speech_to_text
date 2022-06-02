"""
Define functions used to construct a multilayer GRU CTC model, and
functions for training and testing it.
"""

import ctc
import logging
import keras.backend as K

from keras.layers import (BatchNormalization, Convolution1D, Dense,
                          Input, GRU, TimeDistributed)
from keras.models import Model
# from keras.optimizers import SGD
import lasagne

from utils import conv_output_length

logger = logging.getLogger(__name__)