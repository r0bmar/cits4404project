import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from .data_source import DataSource
from .trading_sim import TradingSim, TradingSimV2, Actions
from .market_metrics import MarketMetrics

import matplotlib.pyplot as plt

class PairsTradingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, data_1, data_2, days, spread_status, **kwargs):
        """Initializes the PairsTradingEnv.

        Arguments:
            data_1 {str} -- the stock symbol for Stock 1
            data_2 {str} -- the stock symbol for Stock 2
            days {int} -- the number of days to consider in the observation
                space
            spread_status {int} -- if the spread status is considered or not.

            Example:
                spread_status = 0 : the status is not considered
                spread_status = 1 : the status is considered
                Other values are not legal.

        Key Word Arguments:
            None
        """

        self.spread_status = spread_status
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(days*2+1+self.spread_status,)
        )

        super(PairsTradingEnv, self).__init__()
        self.data_source = DataSource(data_1, data_2, **kwargs)
        self.trading_sim = TradingSim()
        self.market_metrics = MarketMetrics(days)
        self.trading_day = 0
        self.previous_balance = self.trading_sim.balance

        self.render_data = {'buy': [], 'sell': [], 'hold': [], 'portfolio_value': [], 'spread': []}
        self.plot_data = {}

    def seed(self, seed=None):
        """Sets a seed for the envirnoment

        Keyword Arguments:
            seed {any} -- seed value (default: {None})

        Returns:
            [[seed]] -- seed value in array
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the ennvironment

        Returns:
            numpy.Array -- Initial observations from environment
        """
        self.data_source.reset()
        self.trading_sim.reset()
        self.market_metrics.reset()

        self.trading_day = 1
        self.previous_balance = self.trading_sim.balance

        self.render_data = {'buy': [], 'sell': [], 'hold': [], 'portfolio_value': [], 'spread': []}
        self.plot_data = {}

        date, data = next(self.data_source)
        s1_price, s2_price, s1_pct, s2_pct = data

        spread, data_ready = self.market_metrics.update(s1_price, s2_price)
        stock_1_changes, stock_2_changes = self.market_metrics.update_percentage(s1_pct, s2_pct)
        while not data_ready:
            date, data = next(self.data_source)
            s1_price, s2_price, s1_pct, s2_pct = data

            spread, data_ready = self.market_metrics.update(s1_price, s2_price)
            self.trading_day += 1

        if self.spread_status == 0:
            obs = np.array(stock_1_changes+stock_2_changes+[spread])
        else:
            obs = np.array(stock_1_changes+stock_2_changes+[spread, self.trading_sim.status.value])
        return obs

    def skip_forward(self, days):
        """Skip forward a number of days in the envirnoment. Will fail if end of dataset.

        Arguments:
            days {int} -- number of days to skip ahead

        Returns:
            bool -- If skip was successful
        """
        try:
            for _ in range(days):
                date, data = next(self.data_source)
                s1_price, s2_price, s1_pct, s2_pct = data
                self.market_metrics.update(s1_price, s2_price)
                self.market_metrics.update_percentage(s1_pct, s2_pct)
                self.previous_balance = self.trading_sim.get_NAV(s1_price, s2_price)
                self.trading_day += 1
            return True
        except StopIteration:
            False

    def step(self, action, penalty):
        """Perform a set in the environment

        Arguments:
            action {int} -- One of Action enum values
            penalty {float} - Percentage of decrease of balance when performing
                an illegal action.

            Example:
                penalty = 0.9 : the portfolio value is rescaled by 0.9 when
                    an illegal action is performed.

        Returns:
            tuple -- (obs, reward, done, info) implementation of gym
        """
        done = 0
        try:
            date, data = next(self.data_source)
        except StopIteration:
            done = 1
            obs = [0, 0, 0, 0]
            reward = 0
            return obs, reward, done, {}
        s1_price, s2_price, s1_pct, s2_pct = data

        spread, _ = self.market_metrics.update(s1_price, s2_price)
        stock_1_changes, stock_2_changes = self.market_metrics.update_percentage(s1_pct, s2_pct)

        self.trading_sim.execute(action, spread, s1_price, s2_price, penalty)

        self.trading_day += 1
        if self.spread_status == 0:
            obs = np.array(stock_1_changes+stock_2_changes+[spread])
        else:
            obs = np.array(stock_1_changes+stock_2_changes+[spread, self.trading_sim.status.value])
        balance = self.trading_sim.get_NAV(s1_price, s2_price)
        reward = balance / self.previous_balance - 1 # Subtract 1 to centre at 0

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
        """Render the current environment

        Keyword Arguments:
            mode {str} -- Which mode to render as (default: {'human'})
        """
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

# Ignore this class, used for testing.
class PairsTradingEnvV2(gym.Env):
    metadata = {'render.modes': ['human', 'console']}

    action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
    observation_space = spaces.Box(
        low=-1,
        high=1,
        shape=(3,)
    )

    def __init__(self, data_1, data_2, **kwargs):
        super(PairsTradingEnvV2, self).__init__()
        self.data_source = DataSource(data_1, data_2, **kwargs)
        self.trading_sim = TradingSimV2(**kwargs)
        self.market_metrics = MarketMetrics()

        self.trading_day = 0
        self.previous_balance = self.trading_sim.get_NAV()

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
        self.previous_reward = self.trading_sim.get_NAV()

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

    def skip_forward(self, days):
        try:
            for _ in range(days):
                next(self.data_source)
                self.trading_day += 1
            return True
        except StopIteration:
            False

    def step(self, action, penalty):
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

        # self.trading_sim.execute(action, spread, s1_price, s2_price)
        self.trading_sim.redistribute(action, s1_price, s2_price)

        self.trading_day += 1

        obs = np.array([s1_pct, s2_pct, spread])
        self.trading_sim.update_values(s1_price, s2_price)
        balance = self.trading_sim.get_NAV()
        reward = balance / self.previous_balance - 1 # Subtract 1 to centre at 0

        # if action == Actions.BUY.value:
        #     self.render_data['buy'].append((self.trading_day, balance))
        # elif action == Actions.SELL.value:
        #     self.render_data['sell'].append((self.trading_day, balance))
        # else:
        #     self.render_data['hold'].append((self.trading_day, balance))
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

    obsRandom = env.reset()
    plt.ion()

    for days in range(250):
        if obsRandom[len(obsRandom)-1] == 0:
            action_rand = np.random.choice([1,2])
        else:
            action_rand = np.random.choice([0,2])

        obsRandom, r, done, msg = env.step(action_rand)

#             action = np.random.choice([1,2])
#             if obsRandom[3] == 1 and action == 1:
#                 action = 0

        if obsRandom[len(obsRandom)-1] == 0:
            action_rand = np.random.choice([1,2])
        else:
            action_rand = np.random.choice([0,2])

        obsRandom, r, done, msg = env.step(action_rand)
        env.render(mode='human')
        if done: break

    # for _ in range(500):
    #     max_stock_dist = 0.8
    #     dist_a = random.uniform(0,max_stock_dist)

    #     action = [dist_a,max_stock_dist-dist_a,1-max_stock_dist]
    #     action = [0.4, 0.4, 0.2]

    #     obs, reward, done, _ = env.step(action)
    #     env.render(mode='human')

    #     if done: break
    plt.show(block=True)
