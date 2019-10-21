import numpy as np

class MarketMetrics(object):
    days = 5
    
    def __init__(self, window_size=20):
        self._window_size = window_size

        self.window_1 = np.zeros(self._window_size)
        self.window_2 = np.zeros(self._window_size)

        self.queue_1 = [0] * self.days
        self.queue_2 = [0] * self.days

        self._i = 0

    def reset(self):
        self.window_1 = np.zeros(self._window_size)
        self.window_2 = np.zeros(self._window_size)

        self.queue_1 = [0] * self.days
        self.queue_2 = [0] * self.days

        self._i = 0

    def update(self, stock_price_1, stock_price_2):
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