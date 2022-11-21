import copy
import random

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        if len(state.shape) == 3:
            sa = torch.cat([state, action], 2)
        else:
            sa = torch.cat([state, action], 1)

        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class MAMC(object):
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
            actor_lr=1e-3,
            critic_lr=1e-3,
            hidden_sizes=[256, 256],
            q_weight=0.1,
            regularization_weight=0.005,
            actor_num=2,
            critic_num=2,
    ):
        self.device = device

        self.action_dim = action_dim
        self.actor_num = actor_num
        self.critic_num = critic_num
        self.actor = []
        self.actor_target = []
        self.actor_optimizer = []
        self.critic = []
        self.critic_target = []
        self.critic_optimizer = []

        for i in range(actor_num):
            self.actor.append(Actor(state_dim, action_dim, max_action, hidden_sizes).to(self.device))
            self.actor_target.append(copy.deepcopy(self.actor[i]))
            self.actor_optimizer.append(torch.optim.Adam(self.actor[i].parameters(), lr=actor_lr))

        for i in range(critic_num):
            self.critic.append(Critic(state_dim, action_dim, hidden_sizes).to(self.device))
            self.critic_target.append(copy.deepcopy(self.critic[i]))
            self.critic_optimizer.append(torch.optim.Adam(self.critic[i].parameters(), lr=critic_lr))

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.q_weight = q_weight
        self.regularization_weight = regularization_weight

    # use to train
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action_list = torch.FloatTensor(0, self.action_dim).to(self.device)
        q_list = torch.FloatTensor(0, 1).to(self.device)

        for i in range(self.actor_num):
            action_t = self.actor[i](state)
            q_t = self.critic[i](state, action_t)
            action_list = torch.cat((action_list, action_t), 0)
            q_list = torch.cat((q_list, q_t), 0)

        action = action_list[torch.max(q_list, 0)[1]]
        return action.cpu().data.numpy().flatten()

    # use to eval
    def select_action_eval(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        action_list = torch.FloatTensor(0, self.action_dim).to(self.device)
        q_list = torch.FloatTensor(0, 1).to(self.device)

        for i in range(self.actor_num):
            action_t = self.actor[i](state)
            q_t = self.critic[i](state, action_t)
            action_list = torch.cat((action_list, action_t), 0)
            q_list = torch.cat((q_list, q_t), 0)

        action = action_list[torch.max(q_list, 0)[1]]
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):

        # select 2 actor-critic to update
        index = [x for x in range(self.actor_num)]
        random.shuffle(index)

        for i in range(2):
            # update index
            update_index = index[i]

            # sample from replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                #
                next_Q_list = torch.FloatTensor(batch_size, 0).to(self.device)
                for j in range(self.actor_num):
                    # get next action from actor_target[j] and add noise
                    next_action_t = self.actor_target[j](next_state)
                    noise = torch.randn(
                        (action.shape[0], action.shape[1]),
                        dtype=action.dtype, layout=action.layout, device=action.device
                    ) * self.policy_noise
                    next_action_t = (next_action_t + noise).clamp(-self.max_action, self.max_action)

                    # action[j] to critic_target[k] get Qvalue
                    # next_Q_t_list store Qvalue of action[j](size: critic num)
                    next_Q_t_list = torch.FloatTensor(batch_size, 0).to(self.device)
                    for k in range(self.critic_num):
                        next_Q_t = self.critic_target[k](next_state, next_action_t)
                        next_Q_t_list = torch.cat((next_Q_t_list, next_Q_t), 1)

                    # get the min Qvalue of action[j] from next_Q_t_list and pushback to next_Q_list
                    next_Q = torch.min(next_Q_t_list, 1, keepdim=True)[0]
                    next_Q_list = torch.cat((next_Q_list, next_Q), 1)

                # get the max and min Qvalue of each action[j]
                next_Q_max_min = torch.max(next_Q_list, 1, keepdim=True)[0]
                next_Q_min_min = torch.min(next_Q_list, 1, keepdim=True)[0]

                # calculate next_Q from max and min Qvalue of each action[j]
                next_Q = self.q_weight * next_Q_min_min + (1 - self.q_weight) * next_Q_max_min
                target_Q = reward + not_done * self.discount * next_Q

                # get the mean of current Qvalue from critic[j](except now update_index)
               #current_Q_list = torch.Tensor(batch_size, 0).to(self.device)
               #for j in range(self.critic_num):
               #    if j == update_index:
               #        continue
               #    current_Q_t = self.critic[j](state, action)
               #    current_Q_list = torch.cat((current_Q_list, current_Q_t), 1)

               #current_Q_mean = torch.mean(current_Q_list, 1)
               #current_Q_mean = torch.reshape(current_Q_mean, (batch_size, 1))

            # current_Q of update_index
            # regularization loss
            current_Q = self.critic[update_index](state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)
                          #self.regularization_weight * F.mse_loss(current_Q, current_Q_mean)

            self.critic_optimizer[update_index].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[update_index].step()

            actor_loss = -self.critic[update_index](state, self.actor[update_index](state)).mean()

            self.actor_optimizer[update_index].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[update_index].step()

            for param, target_param in zip(self.critic[update_index].parameters(), self.critic_target[update_index].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor[update_index].parameters(), self.actor_target[update_index].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        for i in range(self.actor_num):
            torch.save(self.critic[i].state_dict(), filename + "_critic" + str(i))
            torch.save(self.critic_optimizer[i].state_dict(), filename + "_critic_optimizer" + str(i))
            torch.save(self.actor[i].state_dict(), filename + "_actor" + str(i))
            torch.save(self.actor_optimizer[i].state_dict(), filename + "_actor_optimizer" + str(i))

    def load(self, filename):
        for i in range(self.actor_num):
            self.critic[i].load_state_dict(torch.load(filename + "_critic" + str(i)))
            self.critic_optimizer[i].load_state_dict(torch.load(filename + "_critic_optimizer" + str(i)))
            self.actor[i].load_state_dict(torch.load(filename + "_actor" + str(i)))
            self.actor_optimizer[i].load_state_dict(torch.load(filename + "_actor_optimizer" + str(i)))