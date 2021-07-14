import tensorflow as tf
from tensorflow.keras import Model, Sequential

from hanser.models.layers import Linear


class QNet(Model):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = Sequential([
            Linear(state_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, action_dim)]
        )
        self.in_channels = state_dim
        self.out_channels = action_dim

    def call(self, state):
        return self.net(state)  # Q value


class QNetTwin(Model):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = Sequential([
            Linear(state_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
        ])
        self.net_q1 = Sequential([
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, action_dim),
        ])  # Q1 value
        self.net_q2 = Sequential([
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, action_dim),
        ])  # Q2 value
        self.in_channels = state_dim
        self.out_channels = action_dim

    def call(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # two Q values


class Actor(Model):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = Sequential([
            Linear(state_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, action_dim),
        ])
        self.in_channels = state_dim
        self.out_channels = action_dim

    def call(self, state):
        output = self.net(state)
        return tf.tanh(output)

    def get_action(self, state, action_std):
        action = self(state)
        noise = (tf.random.normal(action.shape) * action_std)
        noise = tf.clip_by_value(noise, -0.5, 0.5)
        action = action + noise
        action = tf.clip_by_value(action, -1.0, 1.0)
        return action


class ActorPPO(Model):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = Sequential([
            Linear(state_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, action_dim),
        ])
        self.in_channels = state_dim
        self.out_channels = action_dim

    def call(self, state):
        return self.net(state)


# class ActorSAC(Model):
#     def __init__(self, mid_dim, state_dim, action_dim):
#         super().__init__()
#         self.net_state = Sequential([Linear(state_dim, mid_dim, act='relu'),
#                                      Linear(mid_dim, mid_dim, act='relu')])
#         self.net_a_avg = Sequential([Linear(mid_dim, mid_dim, act='swish'),
#                                      Linear(mid_dim, action_dim)])  # the average of action
#         self.net_a_std = Sequential([Linear(mid_dim, mid_dim, act='swish'),
#                                      Linear(mid_dim, action_dim)])  # the log_std of action
#         self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))
#
#     def call(self, state):
#         tmp = self.net_state(state)
#         output = self.net_a_avg(tmp)
#         return tf.tanh(output)  # action
#
#     def get_action(self, state):
#         t_tmp = self.net_state(state)
#         a_avg = self.net_a_avg(t_tmp)
#         a_std = self.net_a_std(t_tmp).clamp(-16, 2).exp()
#         return torch.normal(a_avg, a_std).tanh()  # re-parameterize
#
#     def get_action_logprob(self, state):
#         t_tmp = self.net_state(state)
#         a_avg = self.net_a_avg(t_tmp)
#         a_std_log = self.net_a_std(t_tmp).clamp(-16, 2)
#         a_std = a_std_log.exp()
#
#         noise = torch.randn_like(a_avg, requires_grad=True)
#         action = a_avg + a_std * noise
#         a_tan = action.tanh()
#
#         delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
#         logprob = a_std_log + self.sqrt_2pi_log + delta
#         logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
#         return a_tan, logprob.sum(1, keepdim=True)


class Critic(Model):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = Sequential([
            Linear(state_dim + action_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, 1)
        ])
        self.in_channels = state_dim + action_dim
        self.out_channels = 1

    def call(self, state, action):
        return self.net(tf.concat((state, action), axis=1))  # q value


class CriticAdv(Model):
    def __init__(self, state_dim, mid_dim):
        super().__init__()
        self.net = Sequential([
            Linear(state_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
            Linear(mid_dim, 1),
        ])
        self.in_channels = state_dim
        self.out_channels = 1

    def call(self, state):
        return self.net(state)  # Q value


class CriticTwin(Model):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = Sequential([
            Linear(state_dim + action_dim, mid_dim, act='relu'),
            Linear(mid_dim, mid_dim, act='relu'),
        ])  # concat(state, action)
        self.net_q1 = Sequential([
            Linear(mid_dim, mid_dim, act='swish'),
            Linear(mid_dim, 1),
        ])  # q1 value
        self.net_q2 = Sequential([
            Linear(mid_dim, mid_dim, act='swish'),
            Linear(mid_dim, 1),
        ])  # q2 value
        self.in_channels = state_dim + action_dim
        self.out_channels = 1

    def call(self, state, action):
        tmp = self.net_sa(tf.concat((state, action), axis=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(tf.concat((state, action), axis=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
