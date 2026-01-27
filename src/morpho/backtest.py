import numpy as np
import pandas as pd

## Deprecated. Move backetester and RL to rhetenor
class Backtester:
    def __init__(self, close_p, open_p, adj_close_p=None, adj_open_p=None, split=None):
        self.close_p = close_p
        self.open_p = open_p
        if adj_close_p == None:
            self.adj_close_p = close_p
        if adj_open_p == None:
            self.adj_open_p = open_p
        if split == None:
            self.split = np.ones_like(self.close_p, dtype=float)
        # if dividend_per_share == None:
        #     dividend_per_share = np.zeros_like(self.close_p, dtype=float)

        self.ret_overnight = (self.adj_open_p[2:] / self.adj_close_p[1:-1])
        self.drift_overnight = (
            self.open_p[2:] / self.close_p[1:-1]) * self.split[2:]
        self.ret_intraday = (self.adj_close_p[2:] / self.adj_open_p[2:])
        self.drift_intraday = (self.adj_close_p[2:] / self.adj_open_p[2:])

    def run(self, position):
        assert position.shape == self.close_p.shape

        # Position calculated
        calc_d2 = position[0:-2]
        calc_d1 = position[1:-1]
        pos_after_yesterday_close = calc_d2

        # Opening trade
        pos_before_open = pos_after_yesterday_close * self.ret_overnight
        pos_after_open = calc_d1
        tvr_open = pos_after_open - pos_before_open
        ret_overnight = np.nansum(
            pos_before_open - pos_after_yesterday_close, axis=1)

        # Closing trade
        pos_before_close = pos_after_open * self.ret_intraday
        pos_after_close = calc_d1
        tvr_close = pos_after_close - pos_before_close
        ret_intraday = np.nansum(pos_before_close - pos_after_open, axis=1)

        turnover = np.nanmean(np.abs(tvr_open), axis=1) + \
            np.nanmean(np.abs(tvr_close), axis=1)
        returns = ret_overnight + ret_intraday

        return returns, turnover
