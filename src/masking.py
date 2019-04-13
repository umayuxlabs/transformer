import tensorflow as tf
import numpy as np


def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence. It ensures that the model
    does not treat padding as the input. The mask indicates where pad value 0
    is present: it outputs a 1 at those locations, and a 0 otherwise.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence. In other words,
    the mask indicates which entries should not be used.

    This means that to predict the third word, only the first and second word will be used.
    Similarly to predict the fourth word, only the first, second and the third word will
    be used and so on.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

