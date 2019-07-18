import numpy as np
import random
from collections import namedtuple, deque

from model import ActorNet, CriticNet

import torch
import torch.nn.functional as F
import torch.optim as optim

import copy


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
#LR = 0.001#5e-4               # learning rate
UPDATE_EVERY = 1        # how often to update the network

NOISE_SIGMA = 0.1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class G_Noise:
    #""Gaussian Noise""

    def __init__(self, size, seed, sigma=0.1):
        """Initialize parameters and noise process."""
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size

    def sample(self):
        return self.sigma * np.random.standard_normal(self.size)



class Agents(): # based on DQN
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,num_agents, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)


        # Actor Network
        self.actor_local = ActorNet(state_size, action_size, seed).to(device)
        self.actor_target = ActorNet(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_local = CriticNet(state_size, action_size, seed).to(device)
        self.critic_target = CriticNet(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Q-Network
        #self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        #self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        # Noise process (Instead of epsilon in DQN) - taken from example
        self.noise = G_Noise((num_agents, action_size), seed,sigma=NOISE_SIGMA)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done,debugFlag=False):
        # Save experience in replay memory (For N agents)
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA,debugF = debugFlag)


    def act(self, state, noiseFlag=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        states = torch.from_numpy(state).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))

        self.actor_local.eval()
        with torch.no_grad():
            for i in range(self.num_agents):
                actions[i, :] = self.actor_local(states[i,:]).cpu().numpy()
        self.actor_local.train()

        if noiseFlag:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

        # Epsilon-greedy action selection
        #if random.random() > eps:
        #    return np.argmax(action_values.cpu().data.numpy())
        #else:
        #    return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma,debugF=False):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor


        Actor should learn the best argmax_a(Q(s,a)
        Critic should learn to expect the Q(s,a) where a is the chosen action by the actor
        """
        states, actions, rewards, next_states, dones = experiences

        #### CRITIC LEARN ####

        # calc a_next and Q(s,a)_next
        action_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, action_next)

        # calc estimated Q(s,a) (one-step boot straping)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        #calc Q(s,a) from critic local (expected)
        Q_local = self.critic_local(states,actions.float())
        critic_loss = F.mse_loss(Q_local, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #### ACTOR LEARN ####
        actions_predict = self.actor_local(states)
        actor_loss = -self.critic_local(states,actions_predict).mean()  ## we expected low value when actions are good, and minus for the learning direction

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


        '''
        # Get max predicted Q values (for next states) from target model
        act_next_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, act_next_local) 
        
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        if debugF:
            #self.qnetwork_target.eval()
            #tmp = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            #self.qnetwork_target.train()
            print(Q_targets_next)
            
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)    
        '''

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)