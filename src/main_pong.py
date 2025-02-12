import torch
from pettingzoo.butterfly import cooperative_pong_v5
from environment.custom_wrapper import BaseWrapper
from environment.custom_parallel_wrapper import BaseParallelWrapper
from pettingzoo.utils import aec_to_parallel, parallel_to_aec
import numpy as np
from utils.agent_factory import create_agent
import matplotlib.pyplot as plt

# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/cooperative_pong/cooperative_pong.py

torch.manual_seed(123)
np.random.seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = cooperative_pong_v5.parallel_env(render_mode="rgb", cake_paddle=False, off_screen_penalty=-10)
# env = BaseParallelWrapper(env)

# print("Resetting environment")
observations, infos = env.reset()
print(observations, infos)
# print("Environment reset complete")

kierroksia = 0
kierrosluku_total = 0
episodes = 0
max_episodes = 5
max_kierrosluku = 10
kierrosluvut = np.zeros(max_episodes)
print(kierrosluvut)


agents = [
    (
        name,
        create_agent('pong_ppo', env, device),
    )
    for name in env.possible_agents
]

print('agents', agents)

actions = {}
probs = {}
values = {}
thetas = {}
agent_dists = {name: [0, 0] for name in env.possible_agents}


while env.agents:
    kierroksia += 1
    kierrosluku_total += 1

    # agents step
    if kierroksia > 1:
        for name, agent in agents:
            with torch.no_grad():
                # agent chooses action
                action, prob, entropy, value, dist = agent.choose_action(observations[name])

                actions[name] = action
                probs[name] = prob
                values[name] = value
                agent_dists[name] = list(dist)
                # print(dist)

    else:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    env.render()

    past_observations = observations
    

    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(actions)
    # print(rewards, terminations, truncations)
    # print(observations['paddle_0'].shape)
    # print(observations['paddle_0'].transpose().shape)

    # agents learn
    if kierroksia > 1:
        for i, (name, agent) in enumerate(agents):
            if kierrosluku_total > agent.t_learning_starts and kierrosluku_total % agent.training_frequency == 0:
                print(name, "tries to learn", kierrosluku_total, episodes)
                loss = agent.learn(kierrosluku_total, kierroksia)
                print("loss", loss)

    # agents store trajectory
    if kierroksia > 1:
        for i, (name, agent) in enumerate(agents):
            agent.remember(
                kierrosluku_total % agent.training_frequency,
                past_observations[name],
                observations[name],
                actions[name],
                rewards[name],
                probs[name],
                values[name]
            )


    # print(terminations, env.agents)
    # print(kierroksia)

    # if terminations/truncations, record the number of kierroksia (metric of success, along with the rewards of course)

    if (terminations['paddle_0'] or terminations['paddle_1']) or kierroksia == max_kierrosluku:
        observations, infos = env.reset()
        kierrosluvut[episodes] = kierroksia
        print(kierrosluvut)
        kierroksia = 0
        episodes += 1

    if episodes == max_episodes:
        break

print("kierrosluvut", kierrosluvut)
plt.plot(kierrosluvut)
plt.savefig('kierrosluvut.jpg')


env.close()