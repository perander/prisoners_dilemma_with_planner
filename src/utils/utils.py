def get_modified_rewards(rewards, planner_actions, agent_names):
    modified_rewards = {
        agent_names[0]: rewards[agent_names[0]] + planner_actions[0],
        agent_names[1]: rewards[agent_names[1]] + planner_actions[1]
    }

    return modified_rewards


def reset_planner_trajectory():
    planner_trajectory = {}

    planner_trajectory["obs"] = None
    planner_trajectory["action"] = None
    planner_trajectory["reward"] = None
    planner_trajectory["prob"] = None
    planner_trajectory["value"] = None

    return planner_trajectory
