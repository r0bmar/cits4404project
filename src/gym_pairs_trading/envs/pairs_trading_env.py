import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from .data_source import DataSource
from .trading_sim import TradingSim, Actions
from .market_metrics import MarketMetrics

import matplotlib.pyplot as plt

class PairsTradingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console']}

    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(
        low=-1,
        high=1,
        shape=(3,)
    )

    def __init__(self, data_1, data_2, **kwargs):
        super(PairsTradingEnv, self).__init__()
        self.data_source = DataSource(data_1, data_2, **kwargs)
        self.trading_sim = TradingSim()
        self.market_metrics = MarketMetrics()

        self.trading_day = 0
        self.previous_balance = self.trading_sim.balance

        self.render_data = {'buy': [], 'sell': [], 'hold': [], 'portfolio_value': [], 'spread': []}
        self.plot_data = {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.data_source.reset()
        self.trading_sim.reset()
        self.market_metrics.reset()

        self.trading_day = 1
        self.previous_reward = self.trading_sim.balance

        self.render_data = {'buy': [], 'sell': [], 'hold': [], 'portfolio_value': [], 'spread': []}
        self.plot_data = {}

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
        balance = self.trading_sim.get_NAV(s1_price, s2_price)
        reward = balance / self.previous_balance

        if action == Actions.BUY.value:
            self.render_data['buy'].append((self.trading_day, balance))
        elif action == Actions.SELL.value:
            self.render_data['sell'].append((self.trading_day, balance))
        else:
            self.render_data['hold'].append((self.trading_day, balance))
        self.render_data['portfolio_value'].append((self.trading_day, balance))
        self.render_data['spread'].append((self.trading_day, spread))

        self.previous_balance = balance
        return obs, reward, done, {"date": date, "trading_day": self.trading_day}

    def render(self, mode='human'):
        if mode=='human':
            # Setting up matplotlib data structures
            fig = self.plot_data.get('portfolio_value_fig') or plt.figure(figsize=(18, 4.5))
            ax  = self.plot_data.get('portfolio_value_ax') or fig.add_subplot(121)
            ax2  = self.plot_data.get('portfolio_value_ax2') or fig.add_subplot(122)

            self.plot_data['portfolio_value_fig'] = fig
            self.plot_data['portfolio_value_ax'] = ax
            self.plot_data['portfolio_value_ax2'] = ax2

            fig.suptitle("Pairs Trading with Machine Learning")

            # Separate x and y data from lists of tuples
            x_data = lambda data: [x[0] for x in data]
            y_data = lambda data: [x[1] for x in data]

            # Plot Portfolio values
            ax.clear()
            ax.set_title('Portfolio Value')
            ax.plot(
                x_data(self.render_data['portfolio_value']),
                y_data(self.render_data['portfolio_value']),
                label='Portfolio value'
            )

            # Plot Actions data
            ax.plot(x_data(self.render_data['buy']), y_data(self.render_data['buy']), '.r', label='Buy')
            ax.plot(x_data(self.render_data['sell']), y_data(self.render_data['sell']), '^g', label='Sell')
            ax.plot(x_data(self.render_data['hold']), y_data(self.render_data['hold']), 'xy', label='Hold')
            ax.legend(loc='best')

            # Plot spread
            ax2.clear()
            ax2.set_title('Stock Price Spread')
            ax2.plot(
                x_data(self.render_data['spread']),
                y_data(self.render_data['spread']),
                color='#9467bd'
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.05)
        elif mode=='console':
            latest_data = self.render_data # [-1]

            trading_day, nav, _, spread = latest_data

            print(f"Trading day: {trading_day}. Portfolio Value: {nav}. Spread: {spread}")
        else:
            print("Invalid render mode")

if __name__=='__main__':
    env = PairsTradingEnv("AAPL","MSFT")

    import random
    import time

    env.reset()
    plt.ion()
    for _ in range(50):
        obs, reward, done, _ = env.step(random.randint(0,2))
        env.render(mode='human')

        if done: break
    plt.show(block=True)
