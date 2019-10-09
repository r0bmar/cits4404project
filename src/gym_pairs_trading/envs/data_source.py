from datetime import timedelta

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

API_KEY = "SDI1P88XKQ3HASF9"
TIME_SERIES_GETTER = TimeSeries(API_KEY, output_format='pandas')

class DataSource(object):
    def __init__(self, data1, data2):     
        self.d1 = pd.read_csv(data1)
        self.d2 = pd.read_csv(data2)

        self.d1 = self._prep_dataframe(self.d1)
        self.d2 = self._prep_dataframe(self.d2)

        self.trading_day = 0

        self.starting_date = max(min(self.d1['Date']), min(self.d2['Date'])) # highest start date
        self.end_date      = min(max(self.d1['Date']), max(self.d2['Date'])) # lowest end date

        self._trading_days = (self.end_date-self.starting_date).days

    def __iter__(self):
        self.trading_day = 0
        return self

    def __next__(self):
        while self.trading_day != self._trading_days:
            date = self.starting_date+timedelta(days=self.trading_day)
            self.trading_day += 1

            _s1 = self.d1[self.d1['Date']==date]
            _s2 = self.d2[self.d2['Date']==date]

            if _s1.shape[0] != 1: continue
            if _s2.shape[0] != 1: continue

            s1 = _s1.iloc[0]
            s2 = _s2.iloc[0]

            # Calculate percentage change in stock price
            s1_percent_change = (s1.Close - s1.Open) / s1.Open
            s2_percent_change = (s2.Close - s2.Open) / s2.Open

            return (date, np.array([s1.Close, s2.Close, s1_percent_change, s2_percent_change], dtype=np.float))
        raise StopIteration

    def _prep_dataframe(self, df):
        df['Date'] = pd.to_datetime(df['Date'])

        df = df[['Date','Open', 'Close']]
        return df

    def reset(self):
        self.trading_day = 0

if __name__=="__main__":
    ds = DataSource("/Users/asafsilman/Documents/School/CITS4404 - AI/cits4404project/data/AAPL.csv", "/Users/asafsilman/Documents/School/CITS4404 - AI/cits4404project/data/EOD-HD.csv")
    for date, data in ds: print(date, data)
    ds.reset()
    print(next(ds))
