import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from model import Model

'''
    Actions:
    |A| -> 4
    
    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

    States:
    |S| -> 37
    
    Agent's velocity, along with ray-based perception of objects around agent's forward direction. 
    
    Rewards:
    +1 is provided for yellow banana and -1 is provided for blue banana.
'''

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

class Agent:
    # Initialize the constants here, create the function approximators and memory replay structure
    def __init__(self, lr, gamma, batch_size, state_size, action_size):
        self.lr = lr
        self.gamma = gamma

        self.state_size = state_size
        self.action_size = action_size

        print(f'State space: {self.state_size}')
        print(f'State space: {self.action_size}')

        seed = 505

        self.behavioral_model = Model(seed, state_size, action_size).cuda()
        self.target_model = Model(seed, state_size, action_size).cuda()

        self.optimizer = optim.Adam(self.behavioral_model.parameters(), lr=self.lr)

        self.batch_size = batch_size
        self.memory = Memory(n=1e5, batch_size=batch_size, seed=seed)

        self.C = 0
        self.nsteps = 4

    def act(self, state, eps):
        return self._greedy_sample(state, eps)

    def play(self, state):
        self.behavioral_model.eval()
        with torch.no_grad():
            state = torch.from_numpy(state).cuda().float().unsqueeze(0)
            return self.behavioral_model(state).argmax(dim=1).item()

    def train(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)

        if self.batch_size < len(self.memory):
            experience = self.memory.sample()
            self._learn(experience)

    # Update target model from the behavioral model
    def _learn(self, experience):
        states, actions, rewards, next_states, dones = experience

        # Double DQN:
        action_values = self.behavioral_model(next_states)
        max_indices = torch.argmax(action_values, dim=1).unsqueeze(1)

        # Pick the values for the indices above:
        targets = rewards + self.gamma * self.target_model(next_states).detach().gather(1, max_indices) * (1.0 - dones)
        # DQN:
        #targets = rewards + self.gamma * self.target_model(next_states).detach().max(1)[0].unsqueeze(1) * (1.0 - dones)

        outputs = self.behavioral_model(states).gather(1, actions)

        loss = F.mse_loss(outputs, targets)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.C += 1
        if self.C % self.nsteps == 0:
            self._soft_update_policy(tau=1e-3)
            self.C = 0

    def _greedy_sample(self, state, eps):
        prob = np.random.uniform(0.0, 1.0)
        if prob < eps:
            return np.random.choice(np.arange(self.action_size))
        else:
            self.behavioral_model.eval()
            with torch.no_grad():
                state = torch.from_numpy(state).cuda().float().unsqueeze(0)
                outputs = self.behavioral_model(state)
                action = torch.argmax(outputs, dim=1).item()
            self.behavioral_model.train()
            return action

    def _soft_update_policy(self, tau):
        for b_param, t_param in zip(self.behavioral_model.parameters(), self.target_model.parameters()):
            t_param.data.copy_(tau * b_param.data + (1.0 - tau) * t_param.data)

class Memory:
    def __init__(self, n, batch_size, seed):
        self.seed = random.seed(seed)
        self.n = n
        self.batch_size = batch_size
        self.ring_buffer = []
        self.current_i = 0

    def add(self, experience):
        if len(self.ring_buffer) < self.n:
            self.ring_buffer.append(experience)
        else:
            self.current_i %= self.n
            self.ring_buffer[int(self.current_i)] = experience
        self.current_i += 1

    def sample(self):
        samples = random.sample(self.ring_buffer, self.batch_size)
        states = torch.FloatTensor([ entry.state for entry in samples ])
        actions = torch.LongTensor([ entry.action for entry in samples ]).unsqueeze(1)
        rewards = torch.FloatTensor([ entry.reward for entry in samples ]).unsqueeze(1)
        next_states = torch.FloatTensor([ entry.next_state for entry in samples ])
        dones = torch.Tensor([ entry.done for entry in samples ]).unsqueeze(1)

        return [ states.cuda(), actions.cuda(), rewards.cuda(), next_states.cuda(), dones.cuda() ]

    def __len__(self):
        return len(self.ring_buffer)
