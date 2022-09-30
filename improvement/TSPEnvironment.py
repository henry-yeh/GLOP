
from improvement import utils
import numpy as np
from improvement.TSPGraph import TSPGraph



class TSPInstanceEnv():
    """
    TSP Environment
    """

    def __init__(self):
        """
        Initiate TSP Environment

        :param torch tensor points: points in 2D shape (seq_len, 2)
        :param int nof_points: seq_len
        """
        super(TSPInstanceEnv, self).__init__()

        self.visualization = None
        self.observation_space = None
        self.action_space = None

    def reset(self, points, tour, T=None):
        """
        Reset the TSP Environment
        """
        self.T = T
        self.points = points
        self.state = np.copy(self.points)
        # set the current step to 0
        self.current_step = 0
        self.n_bad_actions = 0

        # initiate memory
        self.hist_best_distance = []
        self.hist_current_distance = []

        # tour: list with an initial random tour
        self.tour = tour
        # reset_tour: list with the initial tour of points
        self.reset_tour = self.tour.copy()

        # distances: list of lists with all distances for points
        self.distances = utils.calculate_distances(self.state)
        self.distances = np.rint(self.distances*10000)
        self.distances = self.distances.astype(int)

        # state: reorder the points with the random tour before starting
        # this is the initial state
        self.state = self.state[self.tour, :]
        self.best_state = np.copy(self.state)
        # keep_tours: tour for computing distances (invariant to state)
        self.keep_tour = self.tour.copy()

        # tour_distance: distance of the current tour
        self.tour_distance = utils.route_distance(self.keep_tour,
                                                  self.distances)

        # current best: save the initial tour (keep_tour) and distance
        self.current_best_distance = self.tour_distance
        self.current_best_tour = self.keep_tour.copy()

        # before going to the next state tour gets reset
        self.tour = self.reset_tour.copy()

        # update memory
        self.hist_best_distance.append(self.current_best_distance)
        self.hist_current_distance.append(self.tour_distance)

        return self._next_observation(), self.best_state

    def _next_observation(self):
        """
        Next observation of the TSP Environment
        """
        observation = self.state
        return observation

    def step(self, action):
        """
        Next observation of the TSP Environment
        :param torch tensor action: int (a,b) shape: (1, 2)
        """
        self.current_step += 1

        reward = self._take_action(action)
        observation = self._next_observation()
        done = False  # only stop by number of actions
        if self.T is not None:
            self.T -= 1

        return observation, reward, done, self.best_state

    def _take_action(self, action):
        """
        Take action in the TSP Env
        :param torch.tensor action: indices (i, j) where i <= j shape: (1, 2)
        """
        # tour: new reset tour after a 2opt move
        self.tour = utils.swap_2opt(self.tour,
                                    action[0],
                                    action[1])

        # keep_tour: same 2opt move on keep_tour to keep history
        self.new_keep_tour, self.new_tour_distance = utils.swap_2opt_new(self.keep_tour,
                                                              action[0],
                                                              action[1],
                                                              self.tour_distance,
                                                              self.distances)

        self.state = self.state[self.tour, :]
        self.tour_distance = self.new_tour_distance.copy()
        if (self.current_best_distance > self.tour_distance):
            reward = self.current_best_distance - self.tour_distance
            reward = round(min(reward/10000, 1.0), 4)
            self.current_best_distance = self.tour_distance
            self.current_best_tour = self.new_keep_tour.copy()
            self.best_state = np.copy(self.state)

        else:
            reward = 0.0

        # update memory
        self.hist_current_distance.append(self.tour_distance)
        self.hist_best_distance.append(self.current_best_distance)
        self.keep_tour = self.new_keep_tour.copy()

        # before going to the next state tour gets reset
        self.tour = self.reset_tour.copy()

        return reward


    def _render_to_file(self, filename='render.txt'):
        """
        Render experiences to a file

        :param str filename: filename
        """

        file = open(filename, 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Current Tour: {self.keep_tour}\n')
        file.write(f'Best Tour: {self.current_best_tour}\n')
        file.write(f'Best Distance: {self.current_best_distance}\n')

        file.close()

    def render(self, mode='live', window_size=10, time=0, **kwargs):
        """
        Rendering the episode to file or live

        :param str mode: select mode 'live' or 'file'
        :param int window_size: cost window size for the renderer
        :param title mode: title of the rendere graph
        """
        assert mode == 'file' or mode == 'live'
        # Render the environment
        if mode == 'file':
            self._render_to_file(kwargs.get('filename', 'render.txt'))
        if mode == 'live':
            if self.visualization is None:
                self.visualization = TSPGraph(window_size, time)
            if self.current_step >= window_size:
                self.visualization.render(self.current_step,
                                          self.hist_best_distance,
                                          self.hist_current_distance,
                                          self.state,
                                          self.best_state)

    def close(self):
        """
        Close live rendering
        """
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None


class VecEnv():

    def __init__(self, env, n_envs, n_nodes, T=None):
        self.n_envs = n_envs
        self.env = env
        self.n_nodes = n_nodes
        self.env_idx = np.random.choice(self.n_envs)
        self.T = T

    def create_envs(self):

        self.envs = []
        for i in range(self.n_envs):
            self.envs.append(self.env())

    def reset(self, points, T=None):
        self.create_envs()
        observations = np.ndarray((self.n_envs, self.n_nodes, 2))
        best_observations = np.ndarray((self.n_envs, self.n_nodes, 2))
        self.best_distances = np.ndarray((self.n_envs, 1))
        self.distances = np.ndarray((self.n_envs, 1))

        tour = [x for x in range(self.n_nodes)]
        idx = 0
        for env in self.envs:
            observations[idx], best_observations[idx] = env.reset(points[idx],
                                                                  tour,
                                                                  T)
            self.best_distances[idx] = env.current_best_distance
            self.distances[idx] = env.tour_distance
            idx += 1

        self.current_step = 0

        return observations, self.best_distances.copy(), best_observations

    def step(self, actions):

        observations = np.ndarray((self.n_envs, self.n_nodes, 2))
        best_observations = np.ndarray((self.n_envs, self.n_nodes, 2))
        rewards = np.ndarray((self.n_envs, 1))
        dones = np.ndarray((self.n_envs, 1), dtype=bool)

        idx = 0
        for env in self.envs:
            obs, reward, done, best_obs = env.step(actions[idx])
            self.best_distances[idx] = env.current_best_distance
            self.distances[idx] = env.tour_distance
            observations[idx] = obs
            best_observations[idx] = best_obs
            rewards[idx] = reward
            dones[idx] = done
            idx += 1

        self.current_step += 1
        return observations, rewards, dones, \
            self.best_distances.copy(), self.distances.copy(), \
            best_observations

    def render(self, mode='live', window_size=1, time=0, **kwargs):

        env_to_render = self.envs[self.env_idx]
        env_to_render.render(mode, window_size, self.current_step, **kwargs)

    def calc_avg_distance(self):
        return np.mean(self.best_distances)
