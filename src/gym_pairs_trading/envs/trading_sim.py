from enum import Enum
import numpy as np
import math

class Status(Enum):
    INVESTED_IN_SPREAD = 0
    OUT_OF_SPREAD = 1

class Actions(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

class TradingSim(object):
    def __init__(self, start_balance=1000, transaction_fee=10):
        self._start_balance = start_balance
        
        self.transaction_fee = transaction_fee
        self.balance = self._start_balance
        
        self.stock1_balance = 0
        self.stock2_balance = 0

        self.status = Status.OUT_OF_SPREAD
    
    def reset(self):
        self.balance = self._start_balance

        self.stock1_balance = 0
        self.stock2_balance = 0

        self.status = Status.OUT_OF_SPREAD

    def get_NAV(self, stock1_price, stock2_price):
        return self.balance + \
            self.stock1_balance * stock1_price + \
            self.stock2_balance * stock2_price

    def execute(self, action, spread, stock1_price, stock2_price, penalty):
        action = Actions(action)
        if action == Actions.BUY:

            if self.status == Status.INVESTED_IN_SPREAD:
                first = False
                if(penalty != 1):
                    if self.stock1_balance > 0:
                        # sell stock 1
                        first = True
                        self.balance, self.stock1_balance = self.sell(stock1_price, self.stock1_balance)
                    elif self.stock2_balance > 0:
                        # sell stock 2
                        self.balance, self.stock2_balance = self.sell(stock2_price, self.stock2_balance)

                    self.balance = self.balance*penalty

                    if first:
                        self.balance, self.stock1_balance = self.buy(stock1_price)
                    else:
                        self.balance, self.stock2_balance = self.buy(stock2_price)

                return # Cannot invest if already invested

            # Invest in spread
            if spread < 0:
                # buy stock 1
                self.balance, self.stock1_balance = self.buy(stock1_price)
            else:
                # buy stock 2
                self.balance, self.stock2_balance = self.buy(stock2_price)

            self.status = Status.INVESTED_IN_SPREAD
        elif action == Actions.SELL:
            if self.status == Status.OUT_OF_SPREAD:
                self.balance = self.balance*penalty
                return # Cannot sell if not invested

            if self.stock1_balance > 0:
                # sell stock 1
                self.balance, self.stock1_balance = self.sell(stock1_price, self.stock1_balance)
            elif self.stock2_balance > 0:
                # sell stock 2
                self.balance, self.stock2_balance = self.sell(stock2_price, self.stock2_balance)

            self.status = Status.OUT_OF_SPREAD
        elif action == Actions.HOLD:
            return

    def buy(self, stock_price):
        """Calculates maximum amount of stock that can be bought with
        current cash balance. Returns the new cash and stock balance 
        values.
        
        Arguments:
            stock_price {int} -- Stock price
        Returns:
            (int, int) -- Tuple of new cash balance, and new stock balance
        """

        available_cash_to_spend = self.balance - self.transaction_fee

        max_stocks_to_buy = available_cash_to_spend // stock_price

        new_cash_balance = self.balance - \
            (max_stocks_to_buy * stock_price) - \
            self.transaction_fee

        return (new_cash_balance, max_stocks_to_buy)

    def sell(self, stock_price, stock_balance):
        """Calculates cash balance that is returned when a stock is 
        sold. Returns the new cash and stock balance 
        values.
        
        Arguments:
            stock_price {int} -- Stock price
            stock_balance {int} -- Current stock balance
        Returns:
            (int, int) -- Tuple of new cash balance, and new stock balance
        """

        stock_value = stock_balance * stock_price

        new_cash_balance = self.balance + \
            stock_value - \
            self.transaction_fee

        return (new_cash_balance, 0)

class TradingSimV2(object):
    def __init__(self, **kwargs):
        start_balance = kwargs.get('start_balance', 10000)
        transaction_fee = kwargs.get('transaction_fee', 50)
        
        self._values = np.array([0, 0, 1.0], dtype=np.float32)
        self._balances = np.array([0, 0, start_balance], dtype=np.float32)
        
        self.start_balance = start_balance
    
        self.transaction_fee = transaction_fee

    def reset(self):
        self._values = np.array([0, 0, 1.0], dtype=np.float32)
        self._balances = np.array([0, 0, self.start_balance], dtype=np.float32)

    def update_values(self, stock_1_price, stock_2_price):
        self._values[0] = stock_1_price
        self._values[1] = stock_2_price

    def get_NAV(self):
        return np.sum(self._values * self._balances)

    def get_distribution(self):
        return self._values * self._balances / self.get_NAV()

    def get_balance_delta(self, cash_delta, stock_price):
        if cash_delta < 0:
            units = -math.floor(-cash_delta / stock_price)
            return units, cash_delta-units*stock_price, 
        else:
            units = math.floor(cash_delta / stock_price)
            return units, cash_delta-units*stock_price, 

    def redistribute(self, distribution, stock_1_price, stock_2_price):
        assert (1.0 - np.sum(distribution)) < 0.001, "Invalid input"
        
        self.update_values(stock_1_price, stock_2_price)
        current_NAV = self.get_NAV()
        current_distribution = self.get_distribution()

        distribution_delta = distribution - current_distribution
        if np.sum(np.abs(distribution_delta)) < 0.05:
            # Do not redistribute if delta is less that 5%
            print("No distribtion")
            return
        else:
            print("distribute")

        monatary_delta = distribution_delta * current_NAV

        stock_1_delta, cash_1_delta = self.get_balance_delta(monatary_delta[0], stock_1_price)
        stock_2_delta, cash_2_delta = self.get_balance_delta(monatary_delta[1], stock_2_price)

        cash_delta = monatary_delta[2]+cash_1_delta+cash_2_delta
        value_delta = np.array([stock_1_delta, stock_2_delta, cash_delta], dtype=np.float32)
        self._balances += value_delta
        self._balances[2] -= self.transaction_fee*2

        if self._balances[2] < 0:
            raise ValueError("Out of money")


if __name__=='__main__':
    ts = TradingSimV2()
    # print(ts.get_NAV())
    # print(ts.get_distribution())
    ts.redistribute(np.array([0.2, 0.4, 0.4]), 60, 23.5)
    print(ts.get_NAV())
    ts.redistribute(np.array([0.0, 0.9, 0.1]), 60, 23.5)
    print(ts.get_NAV())



