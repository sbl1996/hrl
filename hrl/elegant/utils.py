import tensorflow as tf


def normalize(tensor):
    """
    Normalize a tensor to have zero mean and unit variance.
    """
    eps = 1e-5
    axes = tuple(range(1, len(tensor.shape)))
    mean, variance = tf.nn.moments(tensor, axes=axes, keepdims=True)
    return (tensor - mean) / tf.sqrt(variance + tf.constant(eps, dtype=tensor.dtype))


def categorial_log_prob(action, logits):
    return -tf.keras.losses.sparse_categorical_crossentropy(
        action, logits, from_logits=True)


def categorial_sample(logits, axis=-1):
    u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    return tf.argmax(logits - tf.math.log(-tf.math.log(u)), axis=axis, output_type=tf.int32)


def categorial_entropy(logits, axis=-1):
    all_probs = tf.nn.softmax(logits, axis=axis)
    entropy = tf.reduce_sum(tf.math.log(all_probs) * all_probs, axis=axis)
    return entropy
