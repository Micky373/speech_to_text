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

def compile_train_fn(model, learning_rate=2e-4):
    """ Build the CTC training routine for speech models.
    Args:
        model: A keras model (built=True) instance
    Returns:
        train_fn (theano.function): Function that takes in acoustic inputs,
            and updates the model. Returns network outputs and ctc cost
    """
    logger.info("Building train_fn")
    acoustic_input = model.inputs[0]
    network_output = model.outputs[0]
    output_lens = K.placeholder(ndim=1, dtype='int32')
    label = K.placeholder(ndim=1, dtype='int32')
    label_lens = K.placeholder(ndim=1, dtype='int32')
    network_output = network_output.dimshuffle((1, 0, 2))

    ctc_cost = ctc.cpu_ctc_th(network_output, output_lens,
                              label, label_lens).mean()
    trainable_vars = model.trainable_weights
    # optimizer = SGD(nesterov=True, lr=learning_rate, momentum=0.9,
    #                 clipnorm=100)
    # updates = optimizer.get_updates(trainable_vars, [], ctc_cost)
    trainable_vars = model.trainable_weights
    grads = K.gradients(ctc_cost, trainable_vars)
    grads = lasagne.updates.total_norm_constraint(grads, 100)
    updates = lasagne.updates.nesterov_momentum(grads, trainable_vars,
                                                learning_rate, 0.99)
    train_fn = K.function([acoustic_input, output_lens, label, label_lens,
                           K.learning_phase()],
                          [network_output, ctc_cost],
                          updates=updates)
    return train_fn

def compile_test_fn(model):
    """ Build a testing routine for speech models.
    Args:
        model: A keras model (built=True) instance
    Returns:
        val_fn (theano.function): Function that takes in acoustic inputs,
            and calculates the loss. Returns network outputs and ctc cost
    """
    logger.info("Building val_fn")
    acoustic_input = model.inputs[0]
    network_output = model.outputs[0]
    output_lens = K.placeholder(ndim=1, dtype='int32')
    label = K.placeholder(ndim=1, dtype='int32')
    label_lens = K.placeholder(ndim=1, dtype='int32')
    network_output = network_output.dimshuffle((1, 0, 2))

    ctc_cost = ctc.cpu_ctc_th(network_output, output_lens,
                              label, label_lens).mean()
    val_fn = K.function([acoustic_input, output_lens, label, label_lens,
                        K.learning_phase()],
                        [network_output, ctc_cost])
    return val_fn