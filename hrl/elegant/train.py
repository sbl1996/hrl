import numpy as np

from hrl.elegant.eval import Evaluator

def random_explore(env, buffer, target_step, gamma, reward_scale=1.0):
    action_dim = env.action_dim

    state = env.reset()
    steps = 0
    while steps < target_step:
        action = np.random.randint(action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action)
        buffer.append_buffer(state, other)
        state = env.reset() if done else next_state
    return steps, state


def train_offpolicy(
    agent, env, env_eval, buffer, max_epoch, step_per_epoch,
    batch_size, repeat_times, gamma, eval_times=3, show_gap=10):

    steps, state = random_explore(env, buffer, step_per_epoch, gamma)
    agent.state = state
    agent.update_net(buffer, step_per_epoch, batch_size, repeat_times)
    total_step = steps

    if hasattr(agent, 'cri_target') and agent.cri_target is not None:
        agent.cri_target.set_weights(agent.cri.get_weights())
    if hasattr(agent, 'act_target') and agent.act_target is not None:
        agent.act_target.set_weights(agent.act.get_weights())

    break_steps = max_epoch * step_per_epoch if max_epoch > 0 else 2 ** 20

    evaluator = Evaluator(agent_id=0, env=env_eval, eval_times=eval_times, show_gap=show_gap)

    reach_goal = False
    while total_step < break_steps:
        steps = agent.explore_env(env, buffer, step_per_epoch, gamma)
        total_step += steps
        obj_a, obj_c = agent.update_net(buffer, step_per_epoch, batch_size, repeat_times)

        reach_goal = evaluator.evaluate(agent.act, steps, obj_a, obj_c)
        if reach_goal:
            break
    return evaluator, reach_goal


def train_onpolicy(
    agent, env, env_eval, buffer, max_epoch, step_per_epoch,
    batch_size, repeat_times, gamma, eval_times=5, show_gap=10):
    
    break_step = max_epoch * step_per_epoch if max_epoch > 0 else 2 ** 20

    evaluator = Evaluator(agent_id=0, env=env_eval, eval_times=eval_times, show_gap=show_gap)

    total_step = 0
    reach_goal = False
    while total_step < break_step:
        steps = agent.explore_env(env, buffer, step_per_epoch, gamma)
        total_step += steps
        obj_a, obj_c = agent.update_net(buffer, step_per_epoch, batch_size, repeat_times)

        reach_goal = evaluator.evaluate(agent.act, steps, obj_a, obj_c)
        if reach_goal:
            break
    return evaluator, reach_goal