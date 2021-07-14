from typing import Callable, Union, Optional

import tensorflow as tf
from hanser.losses import smooth_l1_loss

from hrl.common import Transition
from hrl.utils import to_tensor


class DQNPolicy:

    def __init__(
        self,
        model_fn: Callable[[], tf.keras.Model],
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        gamma: float = 0.99,
        grad_clip_value: float = 1.0,
        is_double: bool = False,
    ):
        self.policy_net = model_fn()
        self.target_net = model_fn()

        self.policy_net.build((None, self.policy_net.in_channels))
        self.target_net.build((None, self.target_net.in_channels))
        self.target_net.set_weights(self.policy_net.get_weights())

        self.optimizer = optimizer

        self.eps = tf.Variable(0., dtype=tf.float32)
        self.gamma = gamma
        self.grad_clip_value = grad_clip_value
        self.is_double = is_double

    def set_eps(self, eps: Union[float, tf.Tensor]):
        self.eps.assign(eps)

    @tf.function(experimental_compile=True)
    def select_action(self, obs: tf.Tensor) -> tf.Tensor:
        n = tf.shape(obs)[0]
        logits = self.policy_net(obs, training=False)
        n_actions = tf.shape(logits)[-1]
        actions = tf.argmax(logits, axis=1, output_type=tf.int32)
        return tf.where(
            tf.random.uniform((n,)) < self.eps,
            tf.random.uniform((n,), 0, n_actions, dtype=tf.int32),
            actions)

    def optimize(self, batch: Transition):
        obs = to_tensor(batch.obs)
        action = to_tensor(batch.action)
        obs_next = to_tensor(batch.obs_next)
        reward = to_tensor(batch.reward)
        done = to_tensor(batch.done)
        self._optimize(obs, action, obs_next, reward, done)

    @tf.function(experimental_compile=True)
    def _optimize(self, obs, action, obs_next, reward, done):
        # reward_mean = tf.reduce_mean(reward)
        # reward_std = tf.math.reduce_std(reward)
        # reward = (reward - reward_mean) / reward_std

        if self.is_double:
            next_actions = tf.argmax(self.policy_net(obs_next), axis=1)
            next_Q_values = self.target_net(obs_next)
            next_Q_values = tf.gather(next_Q_values, next_actions, axis=1, batch_dims=1)
        else:
            next_Q_values = tf.reduce_max(self.target_net(obs_next), axis=1)
        done = tf.cast(done, next_Q_values.dtype)
        target_Q_values = (1 - done) * next_Q_values * self.gamma + reward

        with tf.GradientTape() as tape:
            all_Q_values = self.policy_net(obs, training=True)
            Q_values = tf.gather(all_Q_values, action, axis=1, batch_dims=1)

            loss = smooth_l1_loss(target_Q_values, Q_values, reduction='none')
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        if self.grad_clip_value:
            grads = [tf.clip_by_value(g, -self.grad_clip_value, self.grad_clip_value) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def sync_weight(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def save(self, filepath):
        self.policy_net.save_weights(filepath)

    def load(self, filepath):
        self.policy_net.load_weights(filepath)
        self.sync_weight()
