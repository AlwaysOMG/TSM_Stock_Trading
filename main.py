# import packages
import datetime
import time
import torch
# pip install tensorboard
# pip install torch-tb-profiler
# tensorboard --logdir='log'
from torch.utils.tensorboard import SummaryWriter 

# import modules
from environment import DataPreProcessing
from environment import Environment
from model.dqn import DQN

# setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialization
episode = 0
training_episode = 1000*300
logging_episode = 1000*10
data = DataPreProcessing('TSM')
env = Environment(data)
agent = DQN(device)
writer = SummaryWriter(log_dir=f"log/DQN_{datetime.date.today()}_{time.time()}")\
    if training_episode > logging_episode else None

while True:
    episode_terminal = False
    current_state = env.Reset()

    while not episode_terminal:
        # interact with environment
        action = agent.SelectAction(current_state)
        new_state, reward, done = env.Step(action)
        episode_terminal = done
        # memory
        memory = [current_state, action, reward, new_state, done]
        agent.Memory(memory)
        # state transition
        current_state = new_state

    # update parameter
    if episode % agent.update_value_network_frequency == 0\
        and len(agent.memory_buffer) > agent.batch_size:
        agent.UpdateValueNetwork()
    agent.EpsilonDecay()

    # tensorboard logging
    if training_episode > logging_episode:
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Loss', agent.loss, episode)
        writer.add_scalar('Capital', env.capital, episode)

    episode += 1
    if episode == training_episode:
        break
