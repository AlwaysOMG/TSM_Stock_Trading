# import packages
# pip install pandas
# pip install pandas-datareader
# pip install matplotlib
import numpy as np
import pandas as pd      
import pandas_datareader as datareader
import matplotlib.pyplot as plt

# data pre-processing
def DataPreProcessing(stock_name) -> np.ndarray:
    data = datareader.DataReader(stock_name, data_source='yahoo')
    data = data.drop(['Adj Close'], axis=1)
    arr = data.to_numpy(dtype='float32')
    return arr

# build the environment that agent interact
class Environment:
    def __init__(self, data) -> None:
        self.data = data
        # time
        self.terminal_date = self.data.shape[0]
        self.current_date = 5
        self.window_size = 5
        self.timestep = 5
        # stock
        self.initial_capital = 10000
        self.capital = 0
        self.last_capital = 0
        self.inventory = []

    def Observation(self) -> np.ndarray:
        starting_id = self.current_date - self.window_size
        weight = np.arange(1, self.window_size+1)
        windowed_data = self.data[starting_id:self.current_date, :] if starting_id >= 0 \
            else self.data[:self.window_size, :]
        max_high = np.max(windowed_data[:, 0])
        min_low = np.min(windowed_data[:, 1])
        avg_open = np.average(windowed_data[:, 2], weights=weight)
        avg_close = np.average(windowed_data[:, 3], weights=weight)
        avg_vol = np.average(windowed_data[:, 4], weights=weight)
        capital = self.capital
        state = np.array([max_high, min_low, avg_open, avg_close, avg_vol, capital])
        return state

    """
    def Reward(self) -> float:
        net = max(0, self.capital - self.last_capital)
        self.last_capital = self.capital
        return net

    """
    def Reward(self, done) -> float:
        reward = self.capital - self.initial_capital if done\
            else 0
        return reward
    
    def Buy(self) -> None:
        buying_price = self.data[self.current_date][2]
        if self.capital > buying_price:
            self.capital -=  buying_price
            self.inventory.append(buying_price)

    def Sell(self) -> None:
        selling_price = self.data[self.current_date][2]
        while self.inventory:
            self.capital += selling_price
            self.inventory.pop()
            
    def Reset(self) -> np.ndarray:
        self.capital = self.initial_capital
        self.last_capital = self.capital
        state = self.Observation()
        return state

    def Step(self, action) -> np.ndarray:
        if action == 1:
            self.Buy()
        elif action == 2:
            self.Sell()
        self.current_date = min(self.current_date + self.timestep, self.terminal_date)
        done = True if self.current_date >= self.terminal_date else False
        reward = self.Reward(done)
        new_state = self.Observation()
        return new_state, reward, done

    def Render(self) -> None:
        print(self.capital)

if __name__ == '__main__':
    data = DataPreProcessing('TSM')
    env = Environment(data)
    env.Reset()
    env.Step(1)
    env.Render()