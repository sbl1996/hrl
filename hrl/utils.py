import numpy as np
import tensorflow as tf

def to_tensor(xs, dtype=None):
    if tf.is_tensor(xs):
        if dtype is not None:
            xs = tf.cast(xs, dtype)
        return xs
    if dtype is None:
        if (isinstance(xs, np.ndarray) and xs.ndim > 0) or isinstance(xs, (np.ndarray, list, tuple)):
            x = xs[0]
        elif isinstance(xs, np.ndarray):
            x = xs.item()
        else:
            x = xs
        if isinstance(x, (np.bool, bool, np.bool_)):
            dtype = tf.bool
        elif isinstance(x, (np.float32, np.float64, float)):
            dtype = tf.float32
        elif isinstance(x, (np.int32, np.int64, int)):
            dtype = tf.int32
        elif isinstance(x, (np.uint8,)):
            dtype = tf.uint8
        else:
            dtype = tf.float32
    xs = tf.convert_to_tensor(xs, dtype)
    return xs