import time

import numpy as np

import tensorflow as tf

class Evaluator:
    def __init__(self, agent_id, eval_times, show_gap, env):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eval_times = eval_times
        self.target_reward = env.target_reward

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def evaluate_save(self, act, steps, obj_a, obj_c) -> bool:
        reward_list = [get_episode_return(self.env, act) for _ in range(self.eval_times)]
        r_avg = np.mean(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            # act_save_path = f'{self.cwd}/actor.pth'
            # act.save_weights(act_save_path)
            # print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:>8}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")

        reach_goal = bool(self.r_max >= self.target_reward)
        if reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<2}  {self.total_step:>8}  {self.target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")
        return reach_goal


def get_episode_return(env, act):
    max_step = env.max_step

    episode_return = 0.0  # sum of rewards in an episode
    state = env.reset()
    for _ in range(max_step):
        s_tensor = tf.convert_to_tensor(state)[None]
        logits = act(s_tensor)
        action = tf.argmax(logits, axis=1)
        action = action.numpy()[0]
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return episode_return
