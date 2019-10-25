import numpy as np

class MarketMetrics(object):
    """Calculates market metrics, including normalised log price spread, and the last `days` worth of price changes"""

    def __init__(self, days=5, window_size=20):
        """Creates a MarketMetrics instance.

        Arguments:
            days {int} -- days worth of price changes are considered.

        Keyword Arguments:
            window_size {int} -- window size to calculate normalised price (default: {20})
        """
        self.days = days
        self._window_size = window_size

        self.window_1 = np.zeros(self._window_size)
        self.window_2 = np.zeros(self._window_size)

        self.queue_1 = [0] * self.days
        self.queue_2 = [0] * self.days

        self._i = 0

    def reset(self):
        """Resets the market metrics.
        """
        self.window_1 = np.zeros(self._window_size)
        self.window_2 = np.zeros(self._window_size)

        self.queue_1 = [0] * self.days
        self.queue_2 = [0] * self.days

        self._i = 0

    def update(self, stock_price_1, stock_price_2):
        """Updates the normalised log price for a stock, and returns the spread.
        Rnough data need to have been processed to normalise the data.

        Arguments:
            stock_price_1 {float} -- Stock price 1
            stock_price_2 {float} -- Stock price 2

        Returns:
            tuple -- (spread, spread_ready)
        """
        index = self._i % self._window_size

        self.window_1[index] = stock_price_1
        self.window_2[index] = stock_price_2

        max_stock_1 = max(self.window_1)
        max_stock_2 = max(self.window_2)

        self._i += 1

        data_ready = self._i >= self._window_size
        if not data_ready:
            return 0, False
        else:
            normalised_log_stock_1 = np.log(self.window_1 / max_stock_1)
            normalised_log_stock_2 = np.log(self.window_2 / max_stock_2)

            spread = normalised_log_stock_1[index] - normalised_log_stock_2[index]

            return spread, True

    def update_percentage(self, stock_pct_1, stock_pct_2):
        """Updates and returns last `days` of percentage changes in stock prices

        Arguments:
            stock_pct_1 {float} -- Stock price 1
            stock_pct_2 {float} -- Stock price 2

        Returns:
            tuple -- ([percentage changes for stock 1], [percentage changes for stock 2])
        """
        self.queue_1.insert(0, stock_pct_1)
        self.queue_2.insert(0, stock_pct_2)

        self.queue_1.pop()
        self.queue_2.pop()

        return self.queue_1, self.queue_2


if __name__=='__main__':
    mm = MarketMetrics(window_size=5)
    print(mm.update(6, 1))
    print(mm.update(5, 2))
    print(mm.update(4, 3))
    print(mm.update(3, 4))
    print(mm.update(2, 5))
    print(mm.update(1, 6))