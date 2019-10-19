import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

API_KEY = "SDI1P88XKQ3HASF9"
TIME_SERIES_GETTER = TimeSeries(API_KEY, output_format='pandas')

CACHE_DIRECTORY = './data'

class DataSource(object):
    def __init__(self, symbol_1, symbol_2, **kwargs):
        size = kwargs.get('size', 'full')
        cache = kwargs.get('cache_data', True)

        if cache:
            today = str(datetime.today().date())
            self.d1 = self.get_from_cache_or_download_and_cache(symbol_1, today, size)
            self.d2 = self.get_from_cache_or_download_and_cache(symbol_2, today, size)
        else:
            self.d1, _ = TIME_SERIES_GETTER.get_daily_adjusted(symbol_1, outputsize=size)
            self.d2, _ = TIME_SERIES_GETTER.get_daily_adjusted(symbol_2, outputsize=size)

        self.starting_date = max(min(self.d1.index), min(self.d2.index)) # highest start date
        self.end_date      = min(max(self.d1.index), max(self.d2.index)) # lowest end date

        self.current_day = self.starting_date

        self.s1_split_coefficient = 1
        self.s2_split_coefficient = 1

    def __iter__(self):
        self.current_day = self.starting_date

        self.s1_split_coefficient = 1
        self.s2_split_coefficient = 1
        return self

    def __next__(self):
        while self.current_day != self.end_date:
            self.current_day = self.current_day + pd.offsets.Day(1)

            try:
                s1 = self.d1.loc[self.current_day]
                s2 = self.d2.loc[self.current_day]
            except KeyError:
                continue

            self.s1_split_coefficient *= s1['8. split coefficient']
            self.s2_split_coefficient *= s2['8. split coefficient']

            s1_open  = s1['1. open']  * self.s1_split_coefficient
            s1_close = s1['4. close'] * self.s1_split_coefficient

            s2_open  = s2['1. open']  * self.s2_split_coefficient
            s2_close = s2['4. close'] * self.s2_split_coefficient

            # Calculate percentage change in stock price
            s1_percent_change = (s1_close - s1_open) / s1_open
            s2_percent_change = (s2_close - s2_open) / s2_open

            return (self.current_day.date(), np.array([s1_close, s2_close, s1_percent_change, s2_percent_change], dtype=np.float))
        raise StopIteration

    def reset(self):
        self.current_day = self.starting_date
        self.s1_split_coefficient = 1
        self.s2_split_coefficient = 1

    def get_from_cache_or_download_and_cache(self, symbol, date, size):
        file_name = f"{CACHE_DIRECTORY}/{symbol}_{size}_{date}.csv"

        if os.path.exists(file_name):
            data = pd.read_csv(file_name)
            data.index = pd.to_datetime(data.index)

            return data
        else:
            data, _ = TIME_SERIES_GETTER.get_daily_adjusted(symbol, outputsize=size)
            data.to_csv(file_name, index_label=False)
            return data

if __name__=="__main__":
    ds = DataSource("AAPL", "MSFT", size='full', cache=True)
    for date, data in ds: print(date, data)
    ds.reset()
    print(next(ds))
