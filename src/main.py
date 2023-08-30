import sys
import argparse
import logging
from environment.prisoners_dilemma import parallel_env
from utils.utils import get_planner_action_per_agent_actions
from utils.agent_factory import create_agent
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import reset_planner_trajectory, get_modified_rewards

if __name__ == "__main__":    
    torch.manual_seed(123)
    np.random.seed(123)

    torch.autograd.set_detect_anomaly(True)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agent_algorithm", help="Agent algorithm", default="vpg")
    parser.add_argument("-r", "--reward_structure", help="The name of the reward structure", default="pd", choices=["pd", "pd_reverse", "fully_cooperative", "pd_original"])
    
    args = parser.parse_args()

    alg = args.agent_algorithm
    reward_structure = args.reward_structure

    logging.debug(f"Agent algorithm: {alg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    writer = SummaryWriter(f"src/runs/{alg}")

    # hyperparameters
    planner_alg = 'vpg_planner'
    epochs = 1
    episodes = 5  # 5
    n_steps = 1
    max_cycles = 2
    use_planner = True
    reset_agents_every_kth_episode = 1
    n_agents = 2

    env = parallel_env(render_mode='ansi', max_cycles=max_cycles, n_agents=n_agents, reward_structure=reward_structure)

    print(env.possible_agents)

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

    obs = env.reset()
    for episode in range(episodes):

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

            logging.debug(f"obs: {obs}, actions: {actions}, rewards: {modified_rewards}, next_obs: {next_obs}")

            # agents learn
            for i, (name, agent) in enumerate(agents):
                if total_steps > agent.t_learning_starts and total_steps % agent.training_frequency == 0:
                    # print(f"agent {i} learns")
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
                
                writer.add_scalars(f"avg reward per episode {name}", {
                    "env": cum_rewards[name] / (episode + 1),
                    "modified": cum_modified_rewards[name] / (episode + 1),
                    "additional": cum_additional_rewards[name] / (episode + 1)
                }, total_steps)

            # agents store trajectory
            if alg == 'vpg':
                for i, (name, agent) in enumerate(agents):
                    agent.remember(
                        total_steps % agent.training_frequency,
                        obs[name],
                        actions[name],
                        modified_rewards[name]
                    )
            elif alg == 'ppo':
                for i, (name, agent) in enumerate(agents):
                    agent.remember(
                        total_steps % agent.training_frequency,
                        obs[name],
                        next_obs[name],
                        actions[name],
                        modified_rewards[name],
                        probs[name],
                        values[name]
                    )

            # planner learns
            if total_steps > planner.t_learning_starts and total_steps % planner.training_frequency == 0:
                # print("planner learns")
                if planner_alg == 'ppo_planner' or planner_alg == 'vpg_planner':
                    planner_loss = planner.learn(total_steps, n_steps, agents)
                
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
                f"{env.possible_agents[i]}": agent_dists[env.possible_agents[i]][0] for i in range(len(env.possible_agents))
            }, total_steps)

            # log thetas
            writer.add_scalars(f"theta ", {
                f"{env.possible_agents[0]}": thetas[env.possible_agents[0]][0],
                f"{env.possible_agents[1]}": thetas[env.possible_agents[1]][0],
            }, total_steps)

            # log action frequencies
            if total_steps % reset_agents_every_kth_episode == 0:

                avg_planner_action_per_agent_actions = get_planner_action_per_agent_actions(planner_alg, planner, n_agents)

                writer.add_scalars(f"planner rewards for agent actions 0", {
                    "c, c": avg_planner_action_per_agent_actions[0][0],
                    "c, d": avg_planner_action_per_agent_actions[0][1],
                    "d, c": avg_planner_action_per_agent_actions[0][2],
                    "d, d": avg_planner_action_per_agent_actions[0][len(avg_planner_action_per_agent_actions[0])-1]
                }, total_steps)

                writer.add_scalars(f"planner rewards for agent actions 1", {
                    "c, c": avg_planner_action_per_agent_actions[1][0],
                    "c, d": avg_planner_action_per_agent_actions[1][1],
                    "d, c": avg_planner_action_per_agent_actions[1][2],
                    "d, d": avg_planner_action_per_agent_actions[1][len(avg_planner_action_per_agent_actions[0])-1]
                }, total_steps)

            total_steps += 1

        # log p(cooperate)
        logging.info(f"episode {episode}, p(c): {[round(agent_dists[env.possible_agents[i]][0].item(), 2) for i in range(len(env.possible_agents))]}")

    env.close()
    writer.close()