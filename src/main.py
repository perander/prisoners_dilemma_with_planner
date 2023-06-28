import sys
import logging
from environment.prisoners_dilemma import parallel_env
from utils.utils import get_planner_actions_per_agent_actions
from utils.agent_factory import create_agent
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import reset_planner_trajectory, get_modified_rewards
from utils.plotting import plot_planner_q_values

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # agent algorithm
    try:
        alg = sys.argv[1]
    except IndexError:
        alg = "vpg"  # by default use vanilla policy gradient
    
    logging.debug(f"Agent algorithm: {alg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"src/runs/{alg}")

    # hyperparameters
    planner_alg = 'q_planner'
    planner_alg = 'ppo_planner'
    epochs = 2
    episodes = 10000
    n_steps = 300
    max_cycles = 300
    use_planner = True

    env = parallel_env(render_mode='ansi', max_cycles=max_cycles)

    agents = [
        (
            name,
            create_agent(alg, env, device),
        )
        for name in env.possible_agents
    ]
    
    planner = create_agent(planner_alg, env, device)
    
    logging.debug(f"Agents: {agents}")
    logging.debug(f"Planner: {planner}")

    total_steps = 0
    cum_rewards = {name: 0 for name in env.possible_agents}
    cum_modified_rewards = {name: 0 for name in env.possible_agents}
    cum_additional_rewards = {name: 0 for name in env.possible_agents}
    all_agent_actions = {name: [0, 0] for name in env.possible_agents}
    cum_rewards_planner = 0
    agent_dists = {name: [0, 0] for name in env.possible_agents}
    planner_dists = {name: [0 for _ in range(planner.n_actions_per_agent)] for name in env.possible_agents}

    cum_planner_actions_per_agent_actions_0 = torch.zeros((2,2))
    cum_planner_actions_per_agent_actions_1 = torch.zeros((2,2))
    agent_action_frequencies = torch.zeros((2,2))

    for episode in range(episodes):
        obs = env.reset()
        planner_trajectory = reset_planner_trajectory()
        actions = {}
        probs = {}
        values = {}
 
        for step in range(n_steps):
            total_steps += 1
        
            for name, agent in agents:
                with torch.no_grad():
                    action, prob, entropy, value, dist = agent.choose_action(obs[name])

                    actions[name] = action
                    probs[name] = prob
                    values[name] = value
                    agent_dists[name] = list(dist)
                    all_agent_actions[name][action] += 1

            planner_obs = actions

            if planner_alg == "q_planner":
                planner_action = planner.choose_action(planner_obs)
            else:
                planner_action, planner_prob, planner_entropy, planner_value, dists = planner.choose_action(planner_obs)
                planner_dists[env.possible_agents[0]] = dists[0]
                planner_dists[env.possible_agents[1]] = dists[1]


            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            logging.debug(f"obs {obs},\t actions {actions},\t rewards {rewards},\t next obs {next_obs}")

            modified_rewards = get_modified_rewards(rewards, planner_action, env.possible_agents, use_planner)

            for i, (name, agent) in enumerate(agents):
                agent.remember(
                    total_steps % agent.training_frequency,
                    obs[name],
                    next_obs[name],
                    actions[name],
                    # rewards[name],  # debugging
                    modified_rewards[name],
                    probs[name],
                    values[name]
                )

                if total_steps > agent.t_learning_starts and total_steps % agent.training_frequency == 0:
                    loss = agent.learn(total_steps, n_steps)

                    logging.info(f"{name}: episode {episode}, step {step}, total_steps {total_steps}, loss {'{0:.4f}'.format(loss)}, cum rewards, {'{0:.4f}'.format(cum_rewards[name])}")

                    writer.add_scalar(f"loss {name}", loss, total_steps)

                # log rewards
                cum_rewards[name] += rewards[name]
                cum_modified_rewards[name] += modified_rewards[name]
                cum_additional_rewards[name] += planner_action[i].item()

                writer.add_scalars(f"cum rewards {name}", {
                    "env": cum_rewards[name],
                    "modified": cum_modified_rewards[name],
                    "additional": cum_additional_rewards[name]
                }, total_steps)

            planner_trajectory["next_obs"] = planner_obs  # planner's last actions lead to these agent actions (planner_obs = actions)

            if planner_alg == 'ppo_planner':
                planner.remember(
                    step,
                    total_steps % planner.training_frequency,
                    planner_trajectory["obs"],
                    planner_trajectory["next_obs"],
                    planner_trajectory["action"],
                    planner_trajectory["reward"],
                    planner_trajectory["prob"],
                    planner_trajectory["value"]
                )
                planner_trajectory["prob"] = planner_prob  # probs of planner's current action
                planner_trajectory["value"] = planner_value  # value of planner's current action

            planner_trajectory["obs"] = planner_obs  # planner observed a set of actions
            planner_trajectory["action"] = planner_action  # planner chose additional rewards
            planner_trajectory["reward"] = sum(rewards.values())  # planners actions lead to these rewards
            # planner_trajectory["reward"] = sum(planner_action)  # debugging

            if total_steps > planner.t_learning_starts and total_steps % planner.training_frequency == 0:
                if planner_alg == 'ppo_planner':
                    planner_loss = planner.learn(total_steps, n_steps)

                    logging.info(f"planner: episode {episode}, step {step}, total steps {total_steps}, loss {'{0:.4f}'.format(loss)}, cum rewards, {'{0:.4f}'.format(cum_rewards_planner)}")

                    writer.add_scalar(f"loss planner", planner_loss, total_steps)
                
                if planner_alg == 'q_planner':
                    planner_q_table = planner.learn(planner_trajectory["obs"], planner_trajectory["action"], planner_trajectory["reward"], planner_trajectory["next_obs"], total_steps)

                    writer.add_scalar("planner epsilon", planner.epsilon, total_steps)
            
            obs = next_obs

            # log planner reward
            cum_rewards_planner += planner_trajectory["reward"]
            writer.add_scalar("cum rewards planner", cum_rewards_planner, total_steps)
            
            # log p(cooperate)
            writer.add_scalars(f"p(cooperate) ", {
                f"{env.possible_agents[0]}": agent_dists[env.possible_agents[0]][0],
                f"{env.possible_agents[1]}": agent_dists[env.possible_agents[1]][0],
            }, total_steps)

            # log action frequencies
            cum_planner_actions_per_agent_actions_0,cum_planner_actions_per_agent_actions_1, agent_action_frequencies, avg_planner_action_per_agent_actions_0, avg_planner_action_per_agent_actions_1, agent_action_frequencies_normalized = get_planner_actions_per_agent_actions(cum_planner_actions_per_agent_actions_0, cum_planner_actions_per_agent_actions_1, agent_action_frequencies, env.possible_agents, actions, planner_action)
            
            writer.add_scalars(f"avg planner rewards for agent actions 0", {
                "c, c": avg_planner_action_per_agent_actions_0[0][0],
                "c, d": avg_planner_action_per_agent_actions_0[0][1],
                "d, c": avg_planner_action_per_agent_actions_0[1][0],
                "d, d": avg_planner_action_per_agent_actions_0[1][1]
            }, total_steps)

            writer.add_scalars(f"avg planner rewards for agent actions 1", {
                "c, c": avg_planner_action_per_agent_actions_1[0][0],
                "c, d": avg_planner_action_per_agent_actions_1[0][1],
                "d, c": avg_planner_action_per_agent_actions_1[1][0],
                "d, d": avg_planner_action_per_agent_actions_1[1][1]
            }, total_steps)

            writer.add_scalars(f"agent actions frequencies", {
                "c, c": agent_action_frequencies_normalized[0][0],
                "c, d": agent_action_frequencies_normalized[0][1],
                "d, c": agent_action_frequencies_normalized[1][0],
                "d, d": agent_action_frequencies_normalized[1][1]
            }, total_steps)

    env.close()
    writer.close()