# import packages
import numpy as np
import pandas as pd      
import pandas_datareader as datareader
import matplotlib.pyplot as plt

# pip install -U scikit-learn
from sklearn.preprocessing import MinMaxScaler

# pip install gym
# gym_register_dir = r'venv/Lib/site-packages/gym/envs/classic_control/TSM_stock'
import gym
from gym import spaces

# data pre-processing
def DataPreProcessing(stock_name) -> np.ndarray:
    data = datareader.DataReader(stock_name, data_source='stooq')
    arr = data.to_numpy(dtype='float32')
    # normalize the feature of volumn
    volumn_arr = arr[:, -1].reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(volumn_arr)
    scaled_volumn_arr = scaler.transform(volumn_arr)
    arr[:, -1] = scaled_volumn_arr.flatten()
    return arr

# build the environment with gym
class StockEnv(gym.Env):
    def __init__(self) -> None:
        super(StockEnv, self).__init__()
        self.data = DataPreProcessing('TSM')
        # time
        self.terminal_date = self.data.shape[0]
        self.current_date = 0
        self.window_size = 5
        self.timestep = 5
        # stock
        self.initial_capital = 10000
        self.capital = 0
        self.last_capital = 0
        self.inventory = []
        # action space and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=100000, shape=(1, 6), dtype=np.float32)

    def next_observation(self) -> np.ndarray:
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
        state = np.array([max_high, min_low, avg_open, avg_close, avg_vol, capital]).astype('float32')
        return np.reshape(state, (1, 6))

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
    
    def take_action(self, action) -> None:
        if action == 1:
            self.Buy()
        elif action == 2:
            self.Sell()
        else:
            pass

    def done(self) -> bool:
        if self.current_date >= self.terminal_date:
            return True
    
    def Reward(self, done) -> float:
        reward = self.capital - self.initial_capital if done else 0
        return reward
    
    def reset(self) -> np.ndarray:
        self.current_date = 5
        self.capital = self.initial_capital
        self.last_capital = self.capital
        state = self.next_observation()
        return state

    def step(self, action) -> np.ndarray:
        self.take_action(action)
        self.current_date = min(self.current_date + self.timestep, self.terminal_date)
        done = self.done()
        reward = self.Reward(done)
        new_state = self.next_observation()
        return new_state, reward, done, {}

    def render(self, mode='human') -> None:
        print(f"step: {self.current_date}")
        print(f"capital: {self.capital}")

if __name__ == '__main__':
    env = StockEnv()
    env.reset()
    while not env.done():
        env.render()
        env.step(env.action_space.sample())
