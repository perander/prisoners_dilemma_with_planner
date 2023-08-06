import sys
import logging
from environment.prisoners_dilemma import parallel_env
from utils.utils import get_planner_action_per_agent_actions
from utils.agent_factory import create_agent
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import reset_planner_trajectory, get_modified_rewards

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    torch.manual_seed(123)
    np.random.seed(123)

    torch.autograd.set_detect_anomaly(True)
    
    # agent algorithm
    try:
        alg = sys.argv[1]  # ppo/vpg/dummy
    except IndexError:
        alg = "vpg"  # by default use vanilla policy gradient
    
    logging.debug(f"Agent algorithm: {alg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"src/runs/{alg}")

    # hyperparameters
    planner_alg = 'vpg_planner'  # 'ppo_planner', 'q_planner'
    epochs = 1
    episodes = 20000
    n_steps = 1
    max_cycles = 2
    use_planner = True
    reset_agents_every_kth_episode = 1

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
    cum_rewards_planner = 0
    agent_dists = {name: [0, 0] for name in env.possible_agents}
    planner_next_obs = {name: [0,0] for name in env.possible_agents}

    for episode in range(episodes):
        # if episode % reset_agents_every_kth_episode == 0:
        #     for _, agent in agents:
        #         agent.reset()

        obs = env.reset()

        planner_trajectory = reset_planner_trajectory()
        actions = {}
        probs = {}
        values = {}
        thetas = {}
 
        for step in range(n_steps):        
            for name, agent in agents:
                with torch.no_grad():
                    # agent chooses action
                    action, prob, entropy, value, dist, theta = agent.choose_action(obs[name])

                    actions[name] = action
                    probs[name] = prob
                    values[name] = value
                    agent_dists[name] = list(dist)
                    thetas[name] = theta

            planner_obs = actions

            # planner chooses action
            planner_action, planner_meta = planner.choose_action(planner_obs)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # calculate total reward for agents
            modified_rewards = get_modified_rewards(rewards, planner_action, env.possible_agents, use_planner)

            # agents learn
            for i, (name, agent) in enumerate(agents):
                if total_steps > agent.t_learning_starts and total_steps % agent.training_frequency == 0:
                    loss = agent.learn(total_steps, n_steps)

                    # log loss
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

            # agents store trajectory
            for i, (name, agent) in enumerate(agents):
                agent.remember(
                    total_steps % agent.training_frequency,
                    obs[name],
                    actions[name],
                    modified_rewards[name]
                )

            # planner learns
            if total_steps > planner.t_learning_starts and total_steps % planner.training_frequency == 0:
                if planner_alg == 'ppo_planner' or planner_alg == 'vpg_planner':
                    planner_loss = planner.learn(total_steps, n_steps, agents[0][1].lr, agents[1][1].lr, agents)
                
                    # log loss
                    writer.add_scalar(f"loss planner", planner_loss, total_steps)
                
            # planner stores trajectory
            planner.remember(step, total_steps % planner.training_frequency, planner_action, planner_obs, planner_next_obs, rewards, obs, planner_meta)
            
            obs = next_obs
            planner_next_obs = planner_obs

            # log planner reward
            cum_rewards_planner += sum(rewards.values())
            writer.add_scalar("cum rewards planner", cum_rewards_planner, total_steps)
            
            # log p(cooperate)
            writer.add_scalars(f"p(cooperate) ", {
                f"{env.possible_agents[0]}": agent_dists[env.possible_agents[0]][0],
                f"{env.possible_agents[1]}": agent_dists[env.possible_agents[1]][0],
            }, total_steps)

            # log thetas
            writer.add_scalars(f"theta ", {
                f"{env.possible_agents[0]}": thetas[env.possible_agents[0]][0],
                f"{env.possible_agents[1]}": thetas[env.possible_agents[1]][0],
            }, total_steps)

            # log action frequencies
            if total_steps % reset_agents_every_kth_episode == 0:

                avg_planner_action_per_agent_actions_0, avg_planner_action_per_agent_actions_1 = get_planner_action_per_agent_actions(planner_alg, planner)

                writer.add_scalars(f"planner rewards for agent actions 0", {
                    "c, c": avg_planner_action_per_agent_actions_0[0],
                    "c, d": avg_planner_action_per_agent_actions_0[1],
                    "d, c": avg_planner_action_per_agent_actions_0[2],
                    "d, d": avg_planner_action_per_agent_actions_0[3]
                }, total_steps)

                writer.add_scalars(f"planner rewards for agent actions 1", {
                    "c, c": avg_planner_action_per_agent_actions_1[0],
                    "c, d": avg_planner_action_per_agent_actions_1[1],
                    "d, c": avg_planner_action_per_agent_actions_1[2],
                    "d, d": avg_planner_action_per_agent_actions_1[3]
                }, total_steps)

            total_steps += 1

        logging.info(f"episode {episode}, p(c): ({agent_dists[env.possible_agents[0]][0]:.2f},{agent_dists[env.possible_agents[1]][0]:.2f}) ")

    env.close()
    writer.close()