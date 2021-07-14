import numpy as np

from numba import njit

import tensorflow as tf

@tf.function(jit_compile=True)
def compute_tf(reward, mask):
    """compute the excepted discounted episode return"""
    steps = reward.shape[0]
    r_sum = tf.TensorArray(tf.float32, size=steps)  # reward sum
    pre_r_sum = 0.0  # reward sum of previous step
    for i in tf.range(steps - 1, -1, -1):
        pre_r_sum = reward[i] + mask[i] * pre_r_sum
        r_sum = r_sum.write(i, pre_r_sum)
    r_sum = r_sum.stack()
    return r_sum


@njit
def compute_np(reward, mask):
    """compute the excepted discounted episode return"""
    steps = len(reward)
    r_sum = np.empty(steps, dtype=np.float32)  # reward sum
    pre_r_sum = 0  # reward sum of previous step
    for i in range(steps - 1, -1, -1):
        r_sum[i] = reward[i] + mask[i] * pre_r_sum
        pre_r_sum = r_sum[i]
    return r_sum


reward = np.random.normal(size=(4096,)).astype(np.float32)
mask = (np.random.uniform(size=(4096,)) > 0.01).astype(np.float32) * 0.9
compute_np(reward, mask)

reward_t = tf.convert_to_tensor(reward)
mask_t = tf.convert_to_tensor(mask)
compute_tf(reward, mask)