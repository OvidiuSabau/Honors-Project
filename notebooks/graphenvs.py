import morphsim as m
import numpy as np
from gym import spaces

class HalfCheetahGraphEnv(object):
    def __init__(self, config):
        self._config = config
        # Use class of standard designs and change only parameter length
        self._design_param_names = []
        self._library = m.EnvLibrary(m.xml_improved, self._design_param_names)
        self._current_env_idx = 0
        self._current_env = None
        self.set_morphology(self._current_env_idx)
        # XML Standard
        # self.init_sim_params = [
        #     # (0, [0.34 , 0.42]),
        #     (0, [0.49, 0.26, 0.43, 0.44, 0.31, 0.28]),
        #     (1, [0.36, 0.44, 0.38, 0.43]),
        #     (2, [0.34, 0.32, 0.59  , 0.39, 0.22]),
        #     (3, [0.22, 0.48, 0.47, 0.53 , 0.54]),
        # ]

        # XML Improved
        self.init_sim_params = [
            (0, [0.49, 0.26, 0.43, 0.44, 0.31, 0.28]),
            (1, [0.36, 0.44, 0.38, 0.43]),
            (2, [0.34, 0.32, 0.59, 0.39, 0.22]),
            (3, [0.22, 0.48, 0.47, 0.53 , 0.54]),
            (4, [0.20, 0.49, 0.23, 0.43, 0.51, 0.48, 0.37]),
        ]

    def reset(self):
        state = self._current_env.reset()
        return state

    def step(self, actions):
        state, reward, done, info = self._current_env.step(actions)
        # print(state)
        return state, reward, False, info

    def set_design(self, vector):
        assert len(vector) == self.get_design_size()
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        self._current_env.set_design(vector)

    def set_morphology(self, idx):
        assert idx < self._library.get_size() and idx >= 0 and isinstance(idx, int)
        if not self._current_env is None:
            self._current_env.close()
        self._current_env_idx = idx
        self._current_env = self._library.get_env(self._current_env_idx)

        # We assume one noe = one joint = one action
        self.action_space = spaces.Box(-1, 1, shape=[self._current_env.get_graph().get_nmbr_nodes()], dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._current_env.observation_length], dtype=np.float32)
        self._state_mapping = self._current_env.get_state_mapping()

    def get_current_morphology(self):
        return self._current_env_idx

    def get_current_design(self):
        return self._current_env.get_current_design()

    def get_design_size(self):
        return self._current_env.get_design_size()

    def get_design_bounds(self):
        lower_bound, upper_bound = self._current_env.get_design_bounds()
        return lower_bound, upper_bound

    def get_nmbr_morphologies(self):
        return self._library.get_size()

    def get_graph(self):
        return self._current_env.get_graph()

    def get_random_morphology(self):
        return np.random.randint(low=0, high=self._library.get_size())

    def get_random_design(self):
        lower_bound, upper_bound = self.get_design_bounds()
        rand_design = []
        for l,u in zip(lower_bound, upper_bound):
            rand_design.append(np.random.uniform(low=l, high=u))
        return rand_design

    def get_state_mapping(self):
        return self._state_mapping

    def get_node_feature_size(self):
        return self._current_env.get_node_feature_size()


class HalfCheetahLenRotGraphEnv(HalfCheetahGraphEnv):
    def __init__(self, config):
        self._config = config
        # Use class of standard designs and change only parameter length
        self._design_param_names = ['length', 'rotation']
        self._library = m.EnvLibrary(m.xml_improved, self._design_param_names)
        self._current_env_idx = 0
        self._current_env = None
        self.set_morphology(self._current_env_idx)
        self.init_sim_params = [
            (0, [0.49, 0.26, 0.43, 0.44, 0.31, 0.28,
                0.78,  0.62,  0.89,  0.72, -0.95, -0.49]),
            (1, [0.36, 0.44, 0.38, 0.43,
                -0.03, -0.69, -0.68, -0.03]),
            (2, [0.34, 0.32, 0.59, 0.39, 0.22,
                0.78, -0.78, -0.10, -0.09, -0.33]),
            (3, [0.22, 0.48, 0.47, 0.53 , 0.54,
                -0.21, -0.86, -0.20,  0.74,  0.61]),
            (4, [0.20, 0.49, 0.23, 0.43, 0.51, 0.48, 0.37,
                -0.90,  0.83,  0.69,  0.17,  0.69, 0.08, -0.64]),
        ]
