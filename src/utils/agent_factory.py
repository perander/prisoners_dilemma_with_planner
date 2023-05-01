import json
import numpy as np
from model.ppo_agent import Agent as PPOAgent

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
            t_learning_starts=params["t_learning_starts"]
        )
    else:
        raise ValueError(f"Unsupported agent type: {name}")

