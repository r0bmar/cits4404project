import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from data_source import DataSource
from trading_sim import TradingSim
from market_metrics import MarketMetrics

class PairsTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(
        low=-1,
        high=1,
        shape=(3,)
    )

    def __init__(self, data_1, data_2):
        super(PairsTradingEnv, self).__init__()
        self.data_source = DataSource(data_1, data_2)
        self.trading_sim = TradingSim()
        self.market_metrics = MarketMetrics()

        self.trading_day = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.data_source.reset()
        self.trading_sim.reset()
        self.market_metrics.reset()

        self.trading_day = 1

        date, data = next(self.data_source)
        s1_price, s2_price, s1_pct, s2_pct = data

        spread, data_ready = self.market_metrics.update(s1_price, s2_price)
        while not data_ready:
            date, data = next(self.data_source)
            s1_price, s2_price, s1_pct, s2_pct = data

            spread, data_ready = self.market_metrics.update(s1_price, s2_price)
            self.trading_day += 1


        return np.array([s1_pct, s2_pct, spread])

    def step(self, action):
        done = 0 
        try:
            date, data = next(self.data_source)
        except StopIteration:
            done = 1
            obs = [0, 0, 0]
            reward = 0
            return obs, reward, done, {}
        s1_price, s2_price, s1_pct, s2_pct = data

        spread, _ = self.market_metrics.update(s1_price, s2_price)

        self.trading_sim.execute(action, spread, s1_price, s2_price)

        self.trading_day += 1

        obs = np.array([s1_pct, s2_pct, spread])
        reward = self.trading_sim.get_NAV(s1_price, s2_price)

        return obs, reward, done, {"date": date, "trading_day": self.trading_day}

if __name__=='__main__':
    env = PairsTradingEnv(
        "/Users/asafsilman/Documents/School/CITS4404 - AI/cits4404project/data/AAPL.csv",
        "/Users/asafsilman/Documents/School/CITS4404 - AI/cits4404project/data/EOD-HD.csv"
    )

    import random

    print(env.reset())
    for _ in range(500):
        obs, reward, done, _ = env.step(random.randint(1,3))

        if done: break

        print(obs, reward)
