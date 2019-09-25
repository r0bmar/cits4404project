from enum import Enum

class Status(Enum):
    INVESTED_IN_SPREAD = 1
    OUT_OF_SPREAD = 2

class Actions(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

class TradingSim(object):
    def __init__(self, start_balance=10000, transaction_fee=10):
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

    def execute(self, action, spread, stock1_price, stock2_price):
        action = Actions(action)
        if action == Actions.BUY:
            if self.status == Status.INVESTED_IN_SPREAD:
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


if __name__=='__main__':
    ts = TradingSim()
    ts.execute(1, -0.5, 50, 50)
    print(ts.get_NAV(50, 50))
    print(ts.status)

    ts.execute(2, 0.001, 51, 50)
    print(ts.get_NAV(51, 50))
    print(ts.status)
