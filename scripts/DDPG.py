#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import deque
import random
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from graphenvs import HalfCheetahGraphEnv
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
import time
import networkx as nx

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


# In[3]:


class QValue(nn.Module):
    def __init__(
        self,
        input_size_state,
        input_size_action,
        hidden_sizes
    ):
        super(QValue, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size_action + input_size_state
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_sizes[len(hidden_sizes) - 1], 1))
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
class Policy(nn.Module):
    def __init__(
        self,
        input_size_state,
        hidden_sizes,
        output_size
    ):
        super(Policy, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size_state
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(self.input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_sizes[len(hidden_sizes) - 1], output_size))
        self.layers.append(nn.Tanh())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# In[4]:


class DDPGagent:
    def __init__(self, env, q_hidden_sizes=[32, 64, 128, 64, 32], p_hidden_sizes=[32, 64, 128, 64, 32], actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=int(1e6)):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Networks

        self.actor = Policy(self.num_states, p_hidden_sizes, self.num_actions).to(device)
        self.actor_target = Policy(self.num_states, p_hidden_sizes, self.num_actions).to(device)
        self.critic = QValue(self.num_states, self.num_actions, q_hidden_sizes).to(device)
        self.critic_target = QValue(self.num_states, self.num_actions, q_hidden_sizes).to(device)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        lmbda = lambda epoch: 0.8
        self.actor_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.actor_optimizer, lr_lambda=lmbda)
        self.critic_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.actor_optimizer, lr_lambda=lmbda)

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor.forward(state)
        return action.cpu().detach().numpy()
    
    def get_latest_lr(self):
        return self.critic_lr_scheduler.get_last_lr()
    
    def update_lr(self):
        self.critic_lr_scheduler.step()
        self.actor_lr_scheduler.step()
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
    def save_agent_networks(self, prefix):
        torch.save(self.actor, prefix + '-actor.pt')
        torch.save(self.critic, prefix + '-critic.pt')
        
    def load_agent_networks(self, prefix):
        self.actor = torch.load(prefix + '-actor.pt').to(device)
        self.critic = torch.load(prefix + '-critic.pt').to(device)
        self.actor_target = self.actor
        self.critic_target = self.critic


env = HalfCheetahGraphEnv(None)
idx = 0
env.set_morphology(idx)
state = env.reset()

agent = DDPGagent(env, gamma=0.99)
batch_size = 128
rewards = []
avg_rewards = []
learningRates = []
episodeLengths = []
for episode in range(800):
    t0 = time.time()
    state = env.reset()
    episode_reward = 0
    step = 0
    done = False

    for i in range(1000):

        action = agent.get_action(state)

        # Only add noise in every second episode
        if episode % 2 == 0:
            action = np.clip(action + np.random.normal(0, 0.25, env.action_space.shape), -1, 1)

        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)


        state = new_state
        episode_reward += reward
        step += 1

        if done:
            break

    if episode % 50 == 0:
        agent.update_lr()

    print("episode {} in {}s: reward for episode: {} || average reward: {} || episode length: {}\n".format(episode, np.round(time.time() - t0, decimals=1), np.round(episode_reward, decimals=2),
                                                                                                              np.round(np.mean(rewards[-25:]), decimals=2), step))

    episodeLengths.append(step)
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-25:]))
    learningRates.append(agent.get_latest_lr())

fig, ax = plt.subplots()
ax.plot(range(len(rewards)), rewards)
ax.plot(range(len(avg_rewards)), avg_rewards)
ax.set(ylabel='Reward')
ax.legend(["Episode", "Last 25 Avg"])
plt.xlabel('Episode')
plt.suptitle('Half-Cheetah {}'.format(idx))
plt.show()
fig.savefig('{}-halfCheetah-Training.jpg'.format(idx))

agent.save_agent_networks('{}-halfCheetah'.format(idx))