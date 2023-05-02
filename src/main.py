from environment.prisoners_dilemma import parallel_env
from utils.agent_factory import create_agent
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import reset_planner_trajectory, get_modified_rewards

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("src/runs/ppo")

    # hyperparameters
    alg = 'ppo'
    epochs = 2
    episodes = 2000
    n_steps = 300
    max_cycles = 300

    env = parallel_env(render_mode='ansi', max_cycles=max_cycles)

    agents = [
        (
            name,
            create_agent(alg, env, device),
        )
        for name in env.possible_agents
    ]

    planner = create_agent("planner", env, device)

    total_steps = 0
    cum_rewards = {name: 0 for name in env.possible_agents}
    cum_modified_rewards = {name: 0 for name in env.possible_agents}
    cum_additional_rewards = {name: 0 for name in env.possible_agents}
    all_agent_actions = {name: [0, 0] for name in env.possible_agents}
    cum_rewards_planner = 0
    agent_dists = {name: [0, 0] for name in env.possible_agents}
    planner_dists = {name: [0 for _ in range(planner.n_actions_per_agent)] for name in env.possible_agents}

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
            planner_action, planner_prob, planner_entropy, planner_value, dists = planner.choose_action(planner_obs)
            planner
            planner_dists[env.possible_agents[0]] = dists[0]
            planner_dists[env.possible_agents[1]] = dists[1]

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            # print("obs", obs, "actions", actions, "rewards", rewards, "next obs", next_obs)

            modified_rewards = get_modified_rewards(rewards, planner_action, env.possible_agents)
            # print("modified rewards", modified_rewards)

            cum_rewards_planner += sum(rewards.values())

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

                cum_rewards[name] += rewards[name]
                cum_modified_rewards[name] += modified_rewards[name]
                cum_additional_rewards[name] += planner_action[i].item()
            
                if total_steps > agent.t_learning_starts and total_steps % agent.training_frequency == 0:
                    loss = agent.learn(total_steps, n_steps)
                    print(name, ": episode", episode, "step", step, "total steps", total_steps, "loss", "{0:.4f}".format(loss), "cum rewards", "{0:.4f}".format(cum_rewards[name]))

                    writer.add_scalars(f"action probabilities {name}", {
                        "cooperate": agent_dists[name][0],
                        "defect": agent_dists[name][1]
                    }, total_steps)

    
                    writer.add_scalars(f"action probabilities planner for {name}", {
                        f"{i}": planner_dists[name][0][i] for i in range(planner.n_actions_per_agent)
                    }, total_steps)

                    writer.add_scalar(f"loss {name}", loss, total_steps)
                    writer.add_scalars(f"cum rewards {name}", {
                        "env": cum_rewards[name],
                        "modified": cum_modified_rewards[name],
                        "additional": cum_additional_rewards[name]
                    }, total_steps)

                    writer.add_scalars(f"action fractions {name}", {
                        "all time 0": all_agent_actions[name][0]/sum(all_agent_actions[name]),
                        "all time 1": all_agent_actions[name][1]/sum(all_agent_actions[name]),
                    }, total_steps)
            
            planner_trajectory["next_obs"] = planner_obs  # planner's last actions lead to these agent actions (planner_obs = actions)
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

            planner_trajectory["obs"] = planner_obs  # planner observed a set of actions
            planner_trajectory["action"] = planner_action  # planner chose additional rewards
            planner_trajectory["reward"] = sum(rewards.values())  # planner's last actions lead to this amount of agent welfare
            planner_trajectory["prob"] = planner_prob  # probs of planner's current action
            planner_trajectory["value"] = planner_value  # value of planner's current action

            if total_steps > planner.t_learning_starts and total_steps % planner.training_frequency == 0:
                planner_loss = planner.learn(total_steps, n_steps)
                print("planner : episode", episode, "step", step, "total steps", total_steps, "loss", "{0:.4f}".format(planner_loss), "cum rewards", "{0:.4f}".format(cum_rewards_planner))

                writer.add_scalar(f"loss planner", planner_loss, total_steps)
                writer.add_scalar("cum rewards planner", cum_rewards_planner, total_steps)

                    
            obs = next_obs

    env.close()
    writer.close()