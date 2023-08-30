# adapted from https://pettingzoo.farama.org/content/environment_creation/

import functools

import gymnasium
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def get_reward_map(reward_structure: str) -> dict:
    """Given a reward structure, return a dictionary with the rewards.
    
    An individual agent's action belongs to one of these four cases:
    - agent cooperates while all agents cooperate
    - agent defects while all agents defect
    - agent cooperates while some others defect
    - agent defects while some others cooperate

    Args:
        reward_structure (str): name of the reward structure (pd/pd_reverse/fully_cooperative/pd_original)

    Returns:
        dict: rewards for different action combinations
    """
    # c: cooperate, d: defect
    rewards = {}

    if reward_structure == "pd":
        rewards["c_given_all_c"] = 0.25
        rewards["d_given_all_d"] = -0.05
        rewards["c_given_some_d"] = -0.5
        rewards["d_given_some_c"] = 0.5
    elif reward_structure == "pd_reverse":
        rewards["c_given_all_c"] = -0.05
        rewards["d_given_all_d"] = 0.25
        rewards["c_given_some_d"] = 0.5
        rewards["d_given_some_c"] = -0.5
    elif reward_structure == "fully_cooperative":
        rewards["c_given_all_c"] = 0.5
        rewards["d_given_all_d"] = -0.5
        rewards["c_given_some_d"] = 0.5
        rewards["d_given_some_c"] = -0.5
    elif reward_structure == "pd_original":
        rewards["c_given_all_c"] = 3
        rewards["d_given_all_d"] = 1
        rewards["c_given_some_d"] = 0
        rewards["d_given_some_c"] = 4
    
    # what is a good way to end the list if these are all the options, and I want the reward structure names to be visible (and the error has been caught before this method is called)
    # else:
    #     pass

    return rewards


def get_rewards(actions: dict, agent_names: list, reward_map: dict) -> dict:
    """Return rewards for given agent actions.

    A set of actions belongs to one of these three cases:
    - all agents cooperate (number of defections is 0)
    - all agents defect (number of defections is equal to the number of agents)
    - some agents cooperate and some defect (number of defections is more than 0 and less than the number of agents)

    The individual agents' rewards depend on which case the whole set of actions belongs.

    Args:
        actions (dict): agent actions
        agent_names (list): agent names
        reward_map (dict): reward function

    Returns:
        dict: rewards for given actions
    """
    # TODO these rewards are the PD only for 2-3 players. for 4, all c and all d give the same total reward, and for >4, all d is the best

    rewards = {}

    n_defect = sum(actions.values())

    if n_defect > 0 and n_defect < len(agent_names):
        # some cooperate, some defect
        for agent in agent_names:
            if actions[agent] == 0:
                # cooperate
                rewards[agent] = reward_map["c_given_some_d"]
            else:
                # defect
                rewards[agent] = reward_map["d_given_some_c"]
    elif n_defect == 0:
        # all cooperate
        for agent in agent_names:
            rewards[agent] = reward_map["c_given_all_c"]
    else:
        # all defect
        for agent in agent_names:
            rewards[agent] = reward_map["d_given_all_d"]

    return rewards


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None, max_cycles=300, n_agents=2, reward_structure="pd"):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces
        These attributes should not be changed after initialization.
        """
        self.render_mode = render_mode
        self.state = {}  # latest actions(?)
        self.max_cycles = max_cycles
        self.n_agents = n_agents

        self.reward_structure = reward_structure
        self.reward_map = get_reward_map(self.reward_structure)

        self.possible_agents = ["player_" + str(r) for r in range(self.n_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        if len(self.agents) == self.n_agents:
            string = f"Current state: {[f'Agent{i}: {self.state[self.agents[i]]}' for i in len(self.agents)]}"
        else:
            string = "Game over"
        print(string)


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        # observations = {agent: NONE for agent in self.agents}
        observations = {agent: 0 for agent in self.agents}

        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        self.state = actions
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = get_rewards(actions, self.agents, self.reward_map)

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        # print(f"in step: num_moves {self.num_moves}, max_cycles {self.max_cycles}")
        env_truncation = self.num_moves >= self.max_cycles
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is the other player's most recent action
        # observations = {
        #     self.agents[i]: int(actions[self.agents[1 - i]])
        #     for i in range(len(self.agents))
        # }

        # Silva: observation which does not offer context
        observations = {
            self.agents[i]: 0
            for i in range(len(self.agents))
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        # if env_truncation:
        #     print("truncation")
        #     self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos