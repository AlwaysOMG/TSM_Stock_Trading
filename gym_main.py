import numpy as np
import gym

# pip install stable-baselines3
from stable_baselines3 import PPO

env = gym.make('TSMStock-v1')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
