import torch

def get_modified_rewards(rewards, planner_actions, agent_names, use_planner):
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

    return planner_trajectory

def get_planner_actions_per_agent_actions(a, b, agent_action_frequencies, agent_names, actions, planner_action):
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
