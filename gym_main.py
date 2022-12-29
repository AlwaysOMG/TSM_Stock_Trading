# import packages
import datetime
import time
import gym

# pip install tensorboard
# pip install torch-tb-profiler
# tensorboard --logdir='log/stable-baselines3'
from torch.utils.tensorboard import SummaryWriter

# pip install stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN

# setting
algorithm_name = 'DQN'
train_epochs = 1000*100

# train PPO algorithm
env = gym.make('TSMStock-v1')
"""
PPO_model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, \
    tensorboard_log=f"log/{algorithm_name}_{datetime.date.today()}_{time.time()}")
PPO_model.learn(total_timesteps=250*train_epochs)
PPO_model.save('ppo_model')
"""
DQN_model = DQN("MlpPolicy", env, exploration_fraction=0.4, \
    tensorboard_log=f"log/stable-baselines3/{algorithm_name}_{datetime.date.today()}_{time.time()}")
DQN_model.learn(total_timesteps=250*train_epochs)

obs = env.reset()
done = False
while not done:
    action, _states = DQN_model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
