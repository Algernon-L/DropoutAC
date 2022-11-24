import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# TD3ver2
# TD3 + DropoutActor + UCB

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256], dropout_rate=0.01):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], action_dim)
		self.dropout = nn.Dropout(dropout_rate)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.dropout(self.l1(state)))
		a = F.relu(self.dropout(self.l2(a)))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l3 = nn.Linear(hidden_sizes[1], 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
		self.l5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.l6 = nn.Linear(hidden_sizes[1], 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3ucb(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		actor_lr=1e-3,
		critic_lr=1e-3,
		hidden_sizes=[256, 256],
		dropout_rate=0.01,
		ucb_times=1,
	):
		self.device = device

		self.actor = Actor(state_dim, action_dim, max_action, hidden_sizes, dropout_rate).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.actor_target.eval()

		self.critic = Critic(state_dim, action_dim, hidden_sizes).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

		self.ucb_times = ucb_times
		self.action_dim = action_dim


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		action_list = torch.FloatTensor(0, self.action_dim).to(self.device)
		q_list = torch.FloatTensor(0, 1).to(self.device)
		for _ in range(self.ucb_times - 1):
			action_t = self.actor(state)
			action_list = torch.cat((action_list, action_t), 0)
			q1, q2 = self.critic(state, action_t)
			q = torch.cat((q1, q2))
			q_mean = torch.mean(q, 0, keepdim=True)
			q_std = torch.std(q, 0, keepdim=True)
			q_ucb = q_mean + 0.1 * q_std
			q_list = torch.cat((q_list, q_ucb), 0)

		self.actor.eval()
		action_t = self.actor(state)
		action_list = torch.cat((action_list, action_t), 0)
		q1, q2 = self.critic(state, action_t)
		q = torch.cat((q1, q2))
		q_mean = torch.mean(q, 0, keepdim=True)
		q_std = torch.std(q, 0, keepdim=True)
		q_ucb = q_mean + 0.1 * q_std
		q_list = torch.cat((q_list, q_ucb), 0)
		self.actor.train()

		action = action_list[torch.max(q_list, 0)[1]]
		return action.cpu().data.numpy().flatten()

	def select_action_eval(self, state):
		self.actor.eval()
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		self.actor.train()
		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
	
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_it % self.policy_freq == 0:
			self.actor.eval()
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			self.actor.train()
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
