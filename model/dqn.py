# import packages
import numpy as np

# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device) -> None:
        super(FNN,self).__init__()
        self.device = device
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x) -> torch.Tensor:
        #x.to(self.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        y = self.l3(x)
        return y

class DQN:
    def __init__(self, device, training_episode) -> None:
        # training hyperparameter
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.update_value_network_frequency = 5
        # RL hyperparameter
        self.memory_capacity = 1000*1000 # 1000 episodes
        self.discount_rate = 0.9
        self.max_epsilon = 1
        self.min_epsilon = 0.1
        self.update_epsilon_episode = training_episode*0.75
        # attribute
        self.action_dim = 3
        self.epsilon = self.max_epsilon
        self.epsilon_decay_amount = (self.max_epsilon - self.min_epsilon) / self.update_epsilon_episode
        # object
        self.device = device
        self.value_network = FNN(5, 16, 3, device).to(device)
        self.memory_buffer = [] # state, action, reward, next_state, done
        # optimization
        self.loss = 0
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.value_network.parameters(), lr = self.learning_rate)

    def SelectAction(self, state) -> int:
        # exploration and exploitation
        if self.epsilon >= np.random.rand(1)[0]:
            action = np.random.choice(self.action_dim)
        else:
            state = torch.from_numpy(state.astype('float32')).to(self.device)
            action_value = self.value_network(state)
            action = torch.argmax(action_value)
            action = action.cpu().numpy()
        return action

    def UpdateValueNetwork(self) -> None:
        # sample memory
        indices = np.random.choice(len(self.memory_buffer), size=self.batch_size)
        state_sample = np.array([self.memory_buffer[i][0] for i in indices], dtype='float32')
        action_sample = np.array([self.memory_buffer[i][1] for i in indices], dtype='int64')
        reward_sample = np.array([self.memory_buffer[i][2] for i in indices], dtype='float32')
        next_state_sample = np.array([self.memory_buffer[i][3] for i in indices], dtype='float32')
        done_sample = np.array([self.memory_buffer[i][4] for i in indices])
        # array to torch
        state_sample = torch.from_numpy(state_sample).to(self.device)
        action_sample = torch.from_numpy(action_sample).to(self.device)
        reward_sample = torch.from_numpy(reward_sample).to(self.device)
        next_state_sample = torch.from_numpy(next_state_sample).to(self.device)
        done_sample = torch.from_numpy(done_sample).to(self.device)
        # estimate Q value
        action_value = self.value_network(state_sample)
        action_mask = F.one_hot(action_sample, num_classes=self.action_dim)
        y_hat = torch.sum(torch.mul(action_value, action_mask), dim=1)
        # estimeate next Q value
        next_action_value = self.value_network(next_state_sample)
        next_action = torch.argmax(next_action_value, dim=1)
        next_action_mask = F.one_hot(next_action, num_classes=self.action_dim)
        next_q = torch.sum(torch.mul(next_action_value, next_action_mask), dim=1)
        done_mask = done_sample != True
        next_q = torch.mul(next_q, done_mask)
        y = reward_sample + self.discount_rate * next_q
        # update parameter
        loss = self.loss_function(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

    def EpsilonDecay(self) -> None:
        self.epsilon = self.epsilon - self.epsilon_decay_amount\
            if self.epsilon > self.min_epsilon else self.min_epsilon

    def Memory(self, memory) -> None:
        if len(self.memory_buffer) == self.memory_capacity:
            self.memory_buffer.pop(0)
        self.memory_buffer.append(memory)    
