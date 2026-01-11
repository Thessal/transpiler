import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, close_p, open_p):
        self.close_p = close_p
        self.open_p = open_p

        self.ret_overnight = (self.open[2:] / self.close[1:-1]) 
        self.ret_intraday = (self.close[2:] / self.open[2:]) 
          
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
        ret_overnight = np.nansum(pos_before_open - pos_after_yesterday_close, axis=1)

        # Closing trade
        pos_before_close = pos_after_open * self.ret_intraday
        pos_after_close = calc_d1
        tvr_close = pos_after_close - pos_before_close
        ret_intraday = np.nansum(pos_before_close - pos_after_open, axis=1)

        turnover = np.nanmean(np.abs(tvr_open), axis=1) + np.nanmean(np.abs(tvr_close), axis=1)
        returns = ret_overnight + ret_intraday
        
        return returns, turnover