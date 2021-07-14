import gym
import numpy as np


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)
        self.step = self.step_type

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_reward) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_reward = getattr(env, 'target_reward', None)
    target_reward_default = getattr(env.spec, 'reward_threshold', None)
    if target_reward is None:
        target_reward = target_reward_default
    if target_reward is None:
        target_reward = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name: {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step: {max_step} target_reward: {target_reward}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_reward
