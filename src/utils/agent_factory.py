import json
import numpy as np
from model.ppo_agent import Agent as PPOAgent
from model.ppo_planner import Planner

def create_agent(name, env, device):
    with open("hyperparameters.json", "r") as config_file:
        hyperparameters = json.load(config_file)

    if name == "ppo":
        params = hyperparameters["ppo"]
        return PPOAgent(
            n_actions=env.action_space(env.possible_agents[0]).n,
            input_dims=1,
            device=device,
            gamma=params["gamma"],
            lr=params["lr"],
            gae_lambda=params["gae_lambda"],
            policy_clip=params["policy_clip"],
            batch_size=params["batch_size"],
            n_epochs=params["epochs"],
            memory_size=params["training_frequency"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            training_frequency=params["training_frequency"],
            t_learning_starts=params["t_learning_starts"],
            anneal_lr=params["anneal_lr"]
        )
    elif name == "planner":
        params = hyperparameters["planner"]
        n_agents = len(env.possible_agents)

        return Planner(
            n_actions_per_agent=params["n_actions_per_agent"],
            n_agents=n_agents, # TODO is this still needed?
            agent_names=env.possible_agents,
            input_dims=n_agents,
            device=device,
            gamma=params["gamma"],
            lr=params["lr"],
            gae_lambda=params["gae_lambda"],
            policy_clip=params["policy_clip"],
            batch_size=params["batch_size"],
            n_epochs=params["epochs"],
            memory_size=params["training_frequency"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            max_reward=params["max_reward"],
            training_frequency=params["training_frequency"],
            t_learning_starts=params["t_learning_starts"],
            anneal_lr=params["anneal_lr"]
        )
    else:
        raise ValueError(f"Unsupported agent type: {name}")

