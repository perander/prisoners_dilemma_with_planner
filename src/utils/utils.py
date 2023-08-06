import torch

def get_modified_rewards(rewards, planner_actions, agent_names, use_planner):
    # print("planner actions", planner_actions)
    if use_planner:
        modified_rewards = {
            agent_names[0]: rewards[agent_names[0]] + planner_actions[0],
            agent_names[1]: rewards[agent_names[1]] + planner_actions[1]
        }
    else:
        return rewards

    return modified_rewards


def reset_planner_trajectory():
    planner_trajectory = {}

    planner_trajectory["obs"] = None
    planner_trajectory["action"] = None
    planner_trajectory["reward"] = None
    planner_trajectory["prob"] = None
    planner_trajectory["value"] = None
    planner_trajectory["agent_state"] = None

    return planner_trajectory


def get_planner_action_per_agent_actions(planner_alg, planner):
    device = planner.device
    cc = torch.tensor([[0,0]]).to(device)
    cd = torch.tensor([[0,1]]).to(device)
    dc = torch.tensor([[1,0]]).to(device)
    dd = torch.tensor([[1,1]]).to(device)

    actions = []

    for action_pair in [cc, cd, dc, dd]:
        if planner_alg == 'vpg_planner':
            action = planner.actor(action_pair)
            actions.append(action.squeeze(0))
        elif planner_alg == 'ppo_planner':
            dists = planner.actor(action_pair)
            action = torch.stack([dist.sample() for dist in dists])
            # print(f"planner dists for {action_pair}: {[dist.probs for dist in dists]}")
            mapped_action = planner.actions_mapped[action].T[0]
            actions.append(mapped_action)
    
    actions = torch.stack(actions)
    return actions[:,0], actions[:,1]



# def get_planner_action_per_agent_actions(actor, device):
#     cc = torch.tensor([[0,0]]).to(device)
#     cd = torch.tensor([[0,1]]).to(device)
#     dc = torch.tensor([[1,0]]).to(device)
#     dd = torch.tensor([[1,1]]).to(device)

#     actions = []

#     for action_pair in [cc, cd, dc, dd]:
#         action = actor(action_pair)
#         actions.append(action.squeeze(0))

#     actions = torch.stack(actions)
#     return actions[:,0], actions[:,1]


# def get_planner_action_per_agent_actions_sampled(actor, actions_mapped, device):
#     cc = torch.tensor([[0,0]]).to(device)
#     cd = torch.tensor([[0,1]]).to(device)
#     dc = torch.tensor([[1,0]]).to(device)
#     dd = torch.tensor([[1,1]]).to(device)

#     actions = []

#     for action_pair in [cc, cd, dc, dd]:
#         dists = actor(action_pair)
#         action = torch.stack([dist.sample() for dist in dists])
#         # print(f"planner dists for {action_pair}: {[dist.probs for dist in dists]}")
#         mapped_action = actions_mapped[action].T[0]
#         actions.append(mapped_action)
#     print()

#     actions = torch.stack(actions)
#     return actions[:,0], actions[:,1]



def get_planner_actions_per_agent_actions_historical(a, b, agent_action_frequencies, agent_names, actions, planner_action):
    """

    Args:
        a (_type_): A tensor shape (2,2), where e.g. the item [0][1] corresponds to the joint agent action (c,d)
        b (_type_): _description_
        agent_names (_type_): _description_
        actions (_type_): _description_
        planner_action (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print(step+1)
    # print(planner_action)
    # print(actions)
    agent_action_frequencies[actions[agent_names[0]]][actions[agent_names[1]]] += 1
    # print(agent_action_frequencies)

    a[actions[agent_names[0]]][actions[agent_names[1]]] += planner_action[0]
    b[actions[agent_names[0]]][actions[agent_names[1]]] += planner_action[1]
    # print(a)
    # print(b)
    # print(torch.div(a, agent_action_frequencies))
    a_normalized = torch.div(a, agent_action_frequencies)
    b_normalized = torch.div(b, agent_action_frequencies)
    agent_action_frequencies_normalized = torch.div(agent_action_frequencies, torch.sum(agent_action_frequencies))
    # print(torch.div(agent_action_frequencies, (torch.sum(agent_action_frequencies))))
    return a, b, agent_action_frequencies, a_normalized, b_normalized, agent_action_frequencies_normalized
    return a, b
