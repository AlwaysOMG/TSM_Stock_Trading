# import packages
import datetime
import time
import gym

# pip install tensorboard
# pip install torch-tb-profiler
# tensorboard --logdir='log'
from torch.utils.tensorboard import SummaryWriter

# pip install stable-baselines3
from stable_baselines3 import PPO

# setting
algorithm_name = 'PPO'
train_epochs = 1000*100

# train PPO algorithm
env = gym.make('TSMStock-v1')
model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, \
    tensorboard_log=f"log/{algorithm_name}_{datetime.date.today()}_{time.time()}")
model.learn(total_timesteps=250*train_epochs)
model.save('ppo_model')

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
