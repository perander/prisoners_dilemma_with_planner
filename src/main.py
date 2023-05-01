from environment.prisoners_dilemma import parallel_env
from utils.agent_factory import create_agent
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("src/runs/ppo")

    # hyperparameters
    alg = 'ppo'
    epochs = 2
    episodes = 200
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

    total_steps = 0
    cum_rewards = {name: 0 for name in env.possible_agents}
    all_agent_actions = {name: [0, 0] for name in env.possible_agents}

    for episode in range(episodes):
        obs = env.reset()
        
        for step in range(n_steps):
            total_steps += 1
            actions = {}
            probs = {}
            values = {}
        
            for name, agent in agents:
                with torch.no_grad():
                    action, prob, entropy, value = agent.choose_action(obs[name])
                    actions[name] = action
                    probs[name] = prob
                    values[name] = value

                    all_agent_actions[name][action] += 1

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for name, agent in agents:
                agent.remember(
                    total_steps % agent.training_frequency,
                    obs[name],
                    next_obs[name],
                    actions[name],
                    rewards[name],
                    probs[name],
                    values[name]
                )

                cum_rewards[name] += rewards[name]
            
                if total_steps > agent.t_learning_starts and total_steps % agent.training_frequency == 0:
                    loss = agent.learn(total_steps, n_steps)
                    print(name, ": episode", episode, "step", step, "total steps", total_steps, "loss", "{0:.4f}".format(loss), "cum rewards", cum_rewards[name])

                    writer.add_scalar(f"loss {name}", loss, total_steps)
                    writer.add_scalar(f"cum rewards {name}", cum_rewards[name], total_steps)

                    writer.add_scalars(f"action fractions {name}", {
                        "all time 0": all_agent_actions[name][0]/sum(all_agent_actions[name]),
                        "all time 1": all_agent_actions[name][1]/sum(all_agent_actions[name]),
                    }, total_steps)
                    
            obs = next_obs

    env.close()
    writer.close()