# taken from ContinuousControl D4PG

import numpy as np
from collections import namedtuple, deque
import random, copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY_CR = 1e-4    # L2 weight decay CRITIC
WEIGHT_DECAY_AC = 0     # L2 weight decay ACTOR
UPDATE_EVERY = 2        # update target networks every two gradient ascent steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    " Interacts with and learns from the environment "
    
    
    def __init__(self, action_size, state_size, initialize_weights = False,
                 n_agents = 1, random_seed = 123):
        """ Initialize attributes of Agent        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            hidden_layers_actor (list): list of integers, each element is the number of neurons of 
                                        its corresponding Actor hidden layer
            hidden_layers_critic (list): list of integers, each element is the number of neurons of 
                                        its corresponding Critic hidden layer
            initialize_weights (boolean): 
                                        
        """
        
        self.n_agents = n_agents
        self.action_size = action_size
        self.state_size = state_size
        self.seed = random.seed(random_seed)
        
        # self.local_actor = Actor(action_size, state_size, random_seed, hidden_layers = [64, 32], 
        #                          init_weights = initialize_weights)
        # self.target_actor = Actor(action_size, state_size, random_seed, hidden_layers = [64, 32],
        #                           init_weights = initialize_weights)
        self.local_actor = Actor(action_size, state_size, random_seed, hidden_layers = [8, 4], 
                                 init_weights = initialize_weights)
        self.target_actor = Actor(action_size, state_size, random_seed, hidden_layers = [8, 4],
                                  init_weights = initialize_weights)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_AC)

        # self.local_critic = Critic(action_size, state_size, random_seed, hidden_layers = [128, 64, 64, 32], 
        #                            init_weights = initialize_weights)
        # self.target_critic = Critic(action_size, state_size, random_seed, hidden_layers = [128, 64, 64, 32], 
        #                             init_weights = initialize_weights)
        self.local_critic = Critic(action_size, state_size, random_seed, hidden_layers = [16, 16, 8], 
                                   init_weights = initialize_weights)
        self.target_critic = Critic(action_size, state_size, random_seed, hidden_layers = [16, 16, 8], 
                                    init_weights = initialize_weights)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CR)        
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_agents, random_seed)
        
        # list of noise processes, each agent has an independent process
        self.noise = []
        for i in range(n_agents):
            self.noise.append(OUNoise(action_size, random_seed))
        
    def step(self, state, action, reward, next_state, done):
        """ Saves experience in replay memory, and uses random sample from buffer to learn        
        Params
        ======
            state (float ndarray): state of the environment
            action (float ndarray): action chosen by agent
            reward (float ndarray):  reward given by environment after doing this action
            next_state (float ndarray): state of the environment after doing this action
            done (float ndarray): flag indicating if episode has finished after doing this action
        """
        
        self.memory.add(np.expand_dims(state, axis=0), action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE:
            min_learning = len(self.memory) // BATCH_SIZE
            num_learning = np.min([self.n_agents, min_learning])
            
            for i in range(num_learning):
                update_target_net = False
                batch = self.memory.sample()
                if (i + 1) % UPDATE_EVERY == 0:
                    update_target_net = True
                    
                self.learn(batch, GAMMA, update_target_net)
        
        
    def act(self, state, current_score, max_score, add_noise = True):
        """ Given a state choose an action
        Params
        ======
            state (float ndarray): state of the environment        
        """
        
        state = torch.from_numpy(state.copy()).float().to(device)
        
        self.local_actor.eval() # set network on eval mode, this has any effect only on certain modules (Dropout, BatchNorm, etc.)
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
                        
        self.local_actor.train() # set nework on train mode
        if add_noise:
            for i in range(self.n_agents):
                # the closer ihe score gets to the max score the less noise we add
                damping_factor = (max_score - np.min([max_score, current_score])) / max_score 
                # added_noise = self.noise[i].sample() * damping_factor - 0.5
                # print("added nose:", added_noise)
                added_noise_x = np.random.uniform(-1 * damping_factor, 1 * damping_factor)
                added_noise_y = np.random.uniform(-1 * damping_factor, 1 * damping_factor)
                # print("damping factor", damping_factor)
                
                action[i,:] += [added_noise_x, added_noise_y]
        return np.clip(action, -1, 1)
        
    
    def learn(self, batch, gamma, update_target_net = True):
        """ given a batch of experiences, perform gradient ascent on the local networks and soft update on target networks
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
            
        Params
        ======
            batch (tuple of torch tensors from ndarray): (states, actions, rewards, next_states, dones)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = batch
        
        # compute critic loss
        Q_local = self.local_critic(states, actions)
        Q_target_next = self.target_critic(next_states, self.target_actor(next_states))
        Q_target = rewards + gamma * Q_target_next * (1 - dones)
        critic_loss = F.mse_loss(Q_local, Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1)
        self.critic_optimizer.step()
        
        
        # gradient ascent actor
        Q_local = self.local_critic(states, self.local_actor(states))
        actor_loss = - Q_local.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # soft update of target networks
        if(update_target_net):
            self.soft_update(self.local_critic, self.target_critic, TAU)
            self.soft_update(self.local_actor, self.target_actor, TAU)
        
    
    def reset(self):
        for i in range(self.n_agents):
            self.noise[i].reset()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):            
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
class OUNoise:
    """Ornstein-Uhlenbeck noise process to be added to the actions."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        # self.mu = mu * np.zeros(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
    
class ReplayBuffer:
    " Internal memory of the agent "
    
    def __init__(self, buffer_size, batch_size, n_agents, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.n_agents = n_agents
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        " Add a new experience to memory "
        for i in range(self.n_agents):
            e = self.experience(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
            self.memory.append(e)
        
    def sample(self):
        " Randomly sample a batch of experiences from the memory "
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        " Return the current size of internal memory. Overwrites the inherited function len. "
        
        return len(self.memory)
        
    
    

