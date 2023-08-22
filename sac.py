import gym
import argparse
import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer, ListStorage

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
 
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--render', action='store_true')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--reset-interval', type=int, default=100)
parser.add_argument('--buffer-size', type=int, default=512)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-episodes', type=int, default=750)
parser.add_argument('--epsilon', type=float, default=0.8)
args = parser.parse_args()

POL_HID_SIZE = 128
MARKOVIAN_STATES = ['XX', 'CC', 'CD', 'DC', 'DD'] 
STATES_LEN = len(MARKOVIAN_STATES)
EP_LENGTH = 500
RESET_INTERVAL = args.reset_interval
BUFFER_BATCH_SIZE = args.batch_size
BUFFER_LEN = args.buffer_size
NUM_EPISODES = args.num_episodes
EPS_GREEDY_PROB = args.epsilon
LEARNING_RATE = 0.005
GAMMA = 0.9

def __create_net(in_size, hidden_size, out_size):
    """Creates a two-layer fully connected neural net with ReLU before the hidden layer

    Args:
        in_size (_type_): _description_
        hidden_size (_type_): _description_
        out_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    return nn.Sequential(nn.Linear(in_size, hidden_size), 
                         nn.ReLU(),
                        nn.Linear(hidden_size, out_size))

def reset_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class SAC(nn.Module):
    """Implements Soft-Actor-Critic"""
    
    
    def __init__(self, num_states, num_actions):
        super(SAC, self).__init__()
        self.policy_net     = __create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        self.policy_net.add(nn.Softmax(dim=1))
        self.s_value_net    = __create_net(in_size=num_states, hidden_size=128, out_size= 1)
        self.s_value_target = __create_net(in_size=num_states, hidden_size=128, out_size= 1)
        self.sa_value_net_1 = __create_net(in_size=num_states+num_actions, hidden_size=128, out_size= 1)
        self.sa_value_net_2 = __create_net(in_size=num_states+num_actions, hidden_size=128, out_size= 1)
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=BUFFER_LEN))
        self.num_states = num_states
        self.num_actions = num_actions
        reset_weights(self.policy_net)
        reset_weights(self.s_value_net)
        reset_weights(self.sa_value_net_1)
        reset_weights(self.sa_value_net_2)
        
    def forward(self):
      pass
        
    def sample_policy(self, states):
        action_probs = F.softmax(self.policy_net(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

    def greedy_policy(self, states):
        _, _,log_action_probs = self.sample_policy(states)
        greedy_action = torch.argmax(log_action_probs, dim=1, keepdim=True)
        return greedy_action

    def pack_buffer(self, transition_params):
      pass
    
    def update(self):
      #Sample transitions from buffer
      buffer_samples = self.buffer.sample(batch_size=BUFFER_BATCH_SIZE)
      net_state_val_losses = []
      net_stact_val_1_losses = []
      net_stact_val_2_losses = []
      policy_losses = []
      
      for sample in buffer_samples:
        # (np.hstack((rewards, curr_state, next_state, action_t)))
        reward_1, reward_2 = sample[0], sample[1]
        curr_state = sample[2:2+self.num_states]
        next_state = sample[2+self.num_states:2+2*self.num_states]
        buff_action = sample[2+2*self.num_states]
        _, action_probs, log_probs = self.sample_policy(curr_state)

        # Computing state value net loss by Eq 6 
        greedy_action = torch.argmax(log_probs, dim=1, keepdim=True)
        q1_sa_value = self.sa_value_net_1(torch.stack([curr_state, greedy_action])).detach()
        q2_sa_value = self.sa_value_net_2(torch.stack([curr_state, greedy_action])).detach()
        min_sa_value = torch.min(q1_sa_value, q2_sa_value)
        state_val_loss = (self.s_value_net(curr_state) - min_sa_value + log_probs[greedy_action].detach())
        net_state_val_losses.append(state_val_loss)
        
        # Computing state-action value net losses by Eq 9
        stact_val_loss_1 = (self.sa_value_net_1(torch.stack([curr_state, buff_action]))
                            - reward_1 - GAMMA * self.s_value_target(next_state).detach())
        stact_val_loss_2 = (self.sa_value_net_2(torch.stack([curr_state, buff_action]))
                            - reward_1 - GAMMA * self.s_value_target(next_state).detach())
        net_stact_val_1_losses.append(stact_val_loss_1)
        net_stact_val_2_losses.append(stact_val_loss_2)
        
        # Computing policy net losses by Eq 9
        policy_loss = log_probs[greedy_action] - min_sa_value.detach()
        policy_losses.append(policy_loss)
      
      torch.stack(net_state_val_losses).sum().backward()
      torch.stack(net_stact_val_1_losses).sum().backward()
      torch.stack(net_stact_val_2_losses).sum().backward()
      torch.stack(policy_losses).sum().backward()
        
            
          
class IPDEnv(gym.Env):
  """Custom Gym Environment to play IPD"""
#   metadata = {'render.modes': ['human']}

  def __init__(self, ep_length=50):
    super(IPDEnv, self).__init__()
    self.action_space = spaces.Discrete(2)
    # Example for using image as input:
    self.curr_t = 0
    self.ep_length = ep_length
    self.observation_space = spaces.MultiBinary(n=5)

  def seed(self, x):
    pass

  def step(self, action: np.ndarray, dry_run=False):
    """Plays one round of prisoners' dilemma.  

      Args:
          action (list): _description_
          dry_run (bool, optional): _description_. Defaults to False.

      Returns:
        observation: Next State (ie. current actions)
        rewards: Numpy array with agent rewards
        done_flag: Indicates if IPD game is over ie. after ep_length iterations
    """
    state = action[0] * 2 + action[1]  # CC=0, CD=1, DC=2, DD=3
    if not dry_run:
        self.curr_t += 1 
    # calculate rewards
    player1_rewards = np.array([-1., -3., 0, -2.])
    player2_rewards = np.array([-1., 0., -3, -2.])
    r1 = player1_rewards[state]
    r2 = player2_rewards[state]
    rewards = np.hstack([r1, r2])

    # calculate observation shape: [4x1] = [player1_cooperated, player1_defected, player2_cooperated, player2_defected]
    observation = np.hstack([0, 
                             (action[0] == 0) & (action[1] == 0), 
                             (action[0] == 0) & (action[1] == 1), 
                             (action[0] == 1) & (action[1] == 0), 
                             (action[0] == 1) & (action[1] == 1)])
    observation = observation.astype(np.float32)
    done_flag = self.curr_t == self.ep_length -1
    return observation, rewards, done_flag, []

    # Execute one time step within the environment
    ...
  def reset(self):
    self.curr_t = 0
    return np.array([1, 0, 0, 0, 0]) 
    # Reset the state of the environment to an initial state

  def render(self, mode='human', close=False):
    pass

def run(num_episodes=NUM_EPISODES, reset_interval=RESET_INTERVAL):
    env = IPDEnv(ep_length=50)
    curr_state, _ = env.reset()
    agent = SAC(num_states=len(curr_state), num_actions=2)
    for ep_num in range(num_episodes):
        ep_buffer = []
        while True:
            action_t = agent.greedy_policy(curr_state)
            action_t = [0, action_t]
            next_state, rewards, is_done = env.step(action_t, dry_run=False)
            reward_1 = rewards[0]
            reward_2 = rewards[1]
            # transition_params = {"reward_1":reward_1, "reward_2":reward_2, "curr_state":curr_state, "next_state":next_state}            
            buffer_t = torch.tensor(np.hstack((rewards, curr_state, next_state, action_t)))
            ep_buffer.append(buffer_t)
            if is_done: break
        agent.buffer.extend(ep_buffer)
    
