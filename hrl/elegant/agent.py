import numpy as np

import tensorflow as tf

from hrl.elegant.utils import categorial_entropy, categorial_log_prob, categorial_sample

class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, on_policy):
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.full = False
        self.action_dim = action_dim  # for self.sample_all
        self.on_policy = on_policy

        other_dim = 1 + 1 + 1 + action_dim if on_policy else 1 + 1 + action_dim
        self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
        self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.full = True
            self.next_idx = 0

    def sample_batch(self, batch_size) -> tuple:
        indices = np.random.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_all(self) -> tuple:
        all_other = self.buf_other[:self.now_len]
        return (all_other[:, 0],  # reward
                all_other[:, 1],  # mask = 0.0 if done else gamma
                all_other[:, 2],  # action
                all_other[:, 3:],  # logits
                self.buf_state[:self.now_len])  # state

    def update_now_len_before_sample(self):
        self.now_len = self.max_len if self.full else self.next_idx

    def empty_buffer_before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.full = False


class AgentBase:
    def __init__(self):
        self.soft_update_tau = 5e-3
        self.criterion = None
        self.state = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None

    def select_action(self, state) -> np.ndarray:
        pass  # return action

    def explore_env(self, env, buffer: ReplayBuffer, target_step, gamma, reward_scale=1.0) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.weights, current_net.weights):
            tar.assign(cur * tau + tar * (1 - tau))


class AgentDQN(AgentBase):
    def __init__(self, critic_fn, cri_optimizer, criterion, explore_rate=0.1):
        super().__init__()
        self.cri = critic_fn()
        self.cri.build((None, self.cri.in_channels))
        self.cri_target = critic_fn()
        self.cri_target.build((None, self.cri_target.in_channels))
        self.act = self.cri

        self.cri_optimizer = cri_optimizer
        self.criterion = criterion

        self.explore_rate = explore_rate  # the probability of choosing action randomly in epsilon-greedy

        self.action_dim = self.cri.out_channels  # chose discrete action randomly in epsilon-greedy

    @tf.function(experimental_compile=True)
    def _select_action(self, state):
        if tf.random.uniform(()) < self.explore_rate:  # epsilon-greedy
            action = tf.random.uniform((), 0, self.action_dim, dtype=tf.int32)
        else:
            action = self.act(state[None])[0]
            action = tf.argmax(action, output_type=tf.int32)
        return action

    def select_action(self, state) -> int:  # for discrete action space
        state = tf.convert_to_tensor(state, tf.float32)
        action = self._select_action(state)
        return action.numpy()

    def get_obj_critic(self, reward, mask, action, state, next_s):
        next_q = self.cri_target(next_s)
        next_q = tf.reduce_max(next_q, axis=1, keepdims=True)
        q_label = reward + mask * next_q

        with tf.GradientTape() as tape:
            q_value = tf.gather(self.cri(state), action, axis=1, batch_dims=1)
            obj_critic = self.criterion(q_label, q_value)
            obj_critic = tf.reduce_mean(obj_critic)
        return tape, obj_critic, q_value

    @tf.function(experimental_compile=True)
    def _update(self, reward, mask, action, state, next_s):
        tape, obj_critic, q_value = self.get_obj_critic(reward, mask, action, state, next_s)
        grads = tape.gradient(obj_critic, self.cri.trainable_variables)
        self.cri_optimizer.apply_gradients(zip(grads, self.cri.trainable_variables))
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critic, q_value

    def update_net(self, buffer: ReplayBuffer, target_step, batch_size, repeat_times=1):
        buffer.update_now_len_before_sample()

        q_value = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            reward, mask, action, state, next_s = map(
                tf.convert_to_tensor, buffer.sample_batch(batch_size))
            action = tf.cast(action, tf.int32)
            obj_critic, q_value = self._update(reward, mask, action, state, next_s)

        return tf.reduce_mean(q_value).numpy(), obj_critic.numpy()


class AgentDoubleDQN(AgentDQN):

    @tf.function(experimental_compile=True)
    def _select_action(self, state):
        logits = self.act(state[None])[0]
        if tf.random.uniform(()) < self.explore_rate:  # epsilon-greedy
            action = categorial_sample(logits)
        else:
            action = tf.argmax(logits, output_type=tf.int32)
        return action

    def get_obj_critic(self, reward, mask, action, state, next_s):
        q1, q2 = self.cri_target.get_q1_q2(next_s)
        next_q = tf.minimum(q1, q2)
        next_q = tf.reduce_max(next_q, axis=1, keepdims=True)
        q_label = reward + mask * next_q

        with tf.GradientTape() as tape:
            q1, q2 = [
                tf.gather(qs, action, axis=1, batch_dims=1)
                for qs in self.act.get_q1_q2(state)
            ]
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            obj_critic = tf.reduce_mean(obj_critic)
        return tape, obj_critic, q1


class AgentPPO(AgentBase):
    def __init__(self, actor_fn, critic_fn, optimizer, criterion,
        ratio_clip=0.25, lambda_entropy=0.01, use_gae=True, lambda_gae=0.95):
        super().__init__()

        self.act = actor_fn()
        self.cri = critic_fn()

        self.act.build((None, self.act.in_channels))
        self.cri.build((None, self.cri.in_channels))

        self.optimizer = optimizer
        self.criterion = criterion
        
        self.ratio_clip = ratio_clip  # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.lambda_entropy = lambda_entropy  # could be 0.01 ~ 0.05
        self.use_gae = use_gae  # if use Generalized Advantage Estimation
        self.lambda_gae = lambda_gae  # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.on_policy = True  # AgentPPO is an on policy DRL algorithm

    @tf.function(experimental_compile=True)
    def _select_action(self, state):
        state = state[None]
        logits = self.act(state)[0]
        action = categorial_sample(logits)
        return action, logits

    def select_action(self, state) -> tuple:
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action, logits = self._select_action(state)
        return action.numpy(), logits.numpy()

    def explore_env(self, env, buffer, target_step, gamma, reward_scale=1.0) -> int:
        buffer.empty_buffer_before_explore()  # necessary for on-policy
        state = env.reset()
        for _ in range(target_step):
            action, logits = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action, *logits)
            buffer.append_buffer(state, other)
            state = env.reset() if done else next_state
        return target_step

    def update_net(self, buffer: ReplayBuffer, target_step, batch_size, repeat_times=4):
        buffer.update_now_len_before_sample()
        assert buffer.now_len == target_step
        buf_reward, buf_mask, buf_action, buf_logits, buf_state = map(
            tf.convert_to_tensor, buffer.sample_all())
        obj_actor, obj_critic = self._update_net(
            buf_reward, buf_mask, buf_action, buf_logits, buf_state, batch_size, repeat_times)
        return obj_actor.numpy(), obj_critic.numpy()

    @tf.function(experimental_compile=True)
    def _update_net(self, buf_reward, buf_mask, buf_action, buf_logits, buf_state, batch_size, repeat_times):
        buf_len = buf_reward.shape[0]

        buf_value = tf.concat([
            self.cri(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)], axis=0)[:, 0]

        if self.use_gae:
            buf_r_sum, buf_advantage = discount_reward_age(buf_reward, buf_mask, buf_value, self.lambda_gae)
        else:
            buf_r_sum = discount_reward(buf_reward, buf_mask)
            buf_advantage = buf_r_sum - buf_mask * buf_value
        buf_advantage = (buf_advantage - tf.reduce_mean(buf_advantage)) / (
            tf.math.reduce_std(buf_advantage) + 1e-5)

        buf_logprob = categorial_log_prob(buf_action, buf_logits)

        obj_actor = tf.zeros(())
        obj_critic = tf.zeros(())
        for _i in tf.range(repeat_times * buf_len // batch_size):
            indices = tf.random.uniform((batch_size,), 0, buf_len, dtype=tf.int32)
            state, action, Q_values, logprob, advantage = map(
                lambda x: tf.gather(x, indices), [buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage])
            obj_actor, obj_critic = self._update_step(state, action, Q_values, logprob, advantage)

        return obj_actor, obj_critic

    def _update_step(self, state, action, r_sum, logprob, advantage):
        with tf.GradientTape() as tape:
            logits = self.act(state)

            new_logprob = categorial_log_prob(action, logits)  # obj_actor
            ratio = tf.exp(new_logprob - logprob)
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * tf.clip_by_value(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = tf.reduce_mean(-tf.minimum(obj_surrogate1, obj_surrogate2))
            
            obj_entropy = categorial_entropy(logits)
            obj_entropy = tf.reduce_mean(obj_entropy)  # policy entropy

            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state)[:, 0]  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(r_sum, value)
            obj_critic = tf.reduce_mean(obj_critic)

            obj_united = obj_actor + obj_critic / (tf.math.reduce_std(r_sum) + 1e-5)

        trainable_variables = self.act.trainable_variables + self.cri.trainable_variables
        grads = tape.gradient(obj_united, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return obj_actor, obj_critic


def discount_reward_age(reward, mask, value, lambda_gae):
    steps = reward.shape[0]
    r_sum = tf.TensorArray(tf.float32, size=steps)  # reward sum
    advantage = tf.TensorArray(tf.float32, size=steps)  # advantage value
    lambda_gae = tf.convert_to_tensor(lambda_gae, tf.float32)

    pre_r_sum = 0.0  # reward sum of previous step
    pre_advantage = 0.0  # advantage value of previous step
    for i in tf.range(steps - 1, -1, -1):
        pre_r_sum = reward[i] + mask[i] * pre_r_sum
        r_sum = r_sum.write(i, pre_r_sum)

        advantage_t = reward[i] + mask[i] * (pre_advantage - value[i])
        advantage = advantage.write(i, advantage_t)
        pre_advantage = value[i] + advantage_t * lambda_gae
    r_sum = r_sum.stack()
    advantage = advantage.stack()
    return r_sum, advantage


def discount_reward(reward, mask):
    steps = reward.shape[0]
    r_sum = tf.TensorArray(tf.float32, size=steps)  # reward sum
    pre_r_sum = 0.0  # reward sum of previous step
    for i in tf.range(steps - 1, -1, -1):
        pre_r_sum = reward[i] + mask[i] * pre_r_sum
        r_sum = r_sum.write(i, pre_r_sum)
    r_sum = r_sum.stack()
    return r_sum