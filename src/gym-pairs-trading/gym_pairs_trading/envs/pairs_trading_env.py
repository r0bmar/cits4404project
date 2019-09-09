import gym
from gym import error, spaces, utils
from gym.utils import seeding

class PairsTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    action_space = spaces.Discrete(3)
    observation_space = None

    def __init__(self):
        self.data_source = None
        self.trading_sim = None
        self.market_metrics = None

        self.trading_day = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.data_source.reset()
        self.trading_sim.reset()
        self.market_metrics.reset()

        self.trading_day = 0

        # TODO:Return initial observation
        return

    def step(self, action):
        pass

