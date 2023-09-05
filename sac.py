import gym
import argparse
import numpy as np
from gym import spaces
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer, ListStorage

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
 
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--render', action='store_true')
parser.add_argument('--log-interval', type=int, default=50)
parser.add_argument('--reset-interval', type=int, default=100)
parser.add_argument('--buffer-size', type=int, default=512)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--agent2', type=int, default=1)
parser.add_argument('--num-episodes', type=int, default=750)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--alr', type=float, default=0.005)
parser.add_argument('--clr', type=float, default=0.005)
args = parser.parse_args()

POL_HID_SIZE = 128
MARKOVIAN_STATES = ['XX', 'CC', 'CD', 'DC', 'DD'] 
STATES_LEN = len(MARKOVIAN_STATES)
EP_LENGTH = 50
RESET_INTERVAL = args.reset_interval
BUFFER_BATCH_SIZE = args.batch_size
BUFFER_LEN = args.buffer_size
NUM_EPISODES = args.num_episodes
EPS_GREEDY_PROB = args.epsilon
LEARNING_RATE = 0.005
GAMMA = 0.95
TAU = 0.005
CRITIC_LR = args.clr
ACTOR_LR  = args.alr
ALPHA = args.alpha
SEED = 42
def a2_policy():
    return args.agent2
 
def create_net(in_size, hidden_size, out_size):
    """Creates a two-layer fully connected neural net with ReLU before the hidden layer

    Args:
        in_size (_type_): _description_
        hidden_size (_type_): _description_
        out_size (_type_): _description_

    Returns:
        nn.Sequential: neural net
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
        self.num_states = num_states
        self.num_actions = num_actions
		
        # Initialize policy net
        self.policy_net     = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        self.policy_net.append(nn.Softmax(dim=-1))
        reset_weights(self.policy_net)

        # Initialize Q nets and copy weights to target
        self.sa_value_net_1         = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        self.sa_value_net_2         = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        self.sa_value_target_net_1  = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        self.sa_value_target_net_2  = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
        reset_weights(self.sa_value_net_1)
        reset_weights(self.sa_value_net_2)
        self.sa_value_target_net_1.load_state_dict(self.sa_value_net_1.state_dict())
        self.sa_value_target_net_2.load_state_dict(self.sa_value_net_2.state_dict())        

        # Initialize alpha
        self.log_alpha = torch.zeros(1, requires_grad=False, device=DEVICE)
        self.alpha = self.log_alpha.exp()
        self.alpha = ALPHA
        #TODO: Skipping alpha optimisation

        #Intialise replay buffer
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=BUFFER_LEN))
        
        #Initialise optimisers
        self.critic_optim_1 = torch.optim.Adam(self.sa_value_net_1.parameters(), lr=CRITIC_LR, eps=1e-4)
        self.critic_optim_2 = torch.optim.Adam(self.sa_value_net_2.parameters(), lr=CRITIC_LR, eps=1e-4)
        self.policy_optim   = torch.optim.Adam(self.policy_net.parameters(), lr=ACTOR_LR, eps=1e-4)

        
    def forward(self):
      pass
        
    def sample_policy(self, states):
        action_probs = self.policy_net(states)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        log_action_probs = torch.log(action_probs + 1e-8)

        return actions, action_probs, log_action_probs

    def greedy_policy(self, states):
        """Samples actions and picks actions with highest log probs. 

        Args:
            states (np.array): State to act at

        Returns:
            int: Index of action to take
        """
        _, _,log_action_probs = self.sample_policy(states)
        greedy_action = torch.argmax(log_action_probs, dim=-1, keepdim=True)
        return greedy_action.detach().item()
    
    def calc_state_val(self, state, target=True):
        _, action_probs, log_action_probs = self.sample_policy(state)
        #TODO: torch min or torch minimum?
        min_state_values = torch.min(self.sa_value_target_net_1(state), self.sa_value_target_net_2(state))
        state_val = (action_probs * (min_state_values - self.alpha * log_action_probs)).sum(dim=-1, keepdim=True)
        # state_val = (action_probs * (state_net(state) - self.alpha * log_action_probs)).sum(dim=1, keepdim=True)
        return state_val
    
    def unpack_buffer(self, sample):
        reward_1 	= sample[0]
        curr_state 	= sample[1:1+self.num_states].float()
        next_state 	= sample[1+self.num_states:1+2*self.num_states].float()
        buff_action = sample[1+2*self.num_states].detach().int().item()
        return reward_1, curr_state, next_state, buff_action

    def update(self):
      #Sample transitions from buffer
      self.critic_optim_1.zero_grad()
      self.critic_optim_2.zero_grad()
      self.policy_optim.zero_grad()
      
      buffer_samples = self.buffer.sample(batch_size=BUFFER_BATCH_SIZE)
      critic_1_losses = []
      critic_2_losses = []
      policy_losses = []
      entropies = []
      
      for sample in buffer_samples:
        # (np.hstack((rewards, curr_state, next_state, action_t)))
        reward_1, curr_state, next_state, buff_action = self.unpack_buffer(sample)

        # Computing Actor loss by minimizing TD Error wrt target
        q1_s_a = self.sa_value_net_1(curr_state)[buff_action]
        q2_s_a = self.sa_value_net_2(curr_state)[buff_action]
        q_s_a_target = reward_1 + GAMMA * self.calc_state_val(next_state)
        critic_1_losses.append(F.mse_loss(q1_s_a, q_s_a_target.detach()))
        critic_2_losses.append(F.mse_loss(q2_s_a, q_s_a_target.detach()))
        
        # Computing policy net losses by Eq 9
        min_s_val = torch.min(self.sa_value_net_1(curr_state), self.sa_value_net_2(curr_state))
        _, action_probs, log_probs = self.sample_policy(curr_state)
        policy_loss = (action_probs * (self.alpha * log_probs - min_s_val.detach())).sum(dim=-1)
        entropy = -(action_probs * log_probs).sum(dim=-1)
        policy_losses.append(policy_loss)
        entropies.append(entropy)
      
      
      critic_1_loss = torch.stack(critic_1_losses).sum()
      critic_2_loss = torch.stack(critic_2_losses).sum()
      policy_loss   = torch.stack(policy_losses).sum()
      mean_entropy  = torch.stack(entropies).mean()
      net_loss = (critic_1_loss + critic_2_loss)/2 + policy_loss
      net_loss.backward()

      self.critic_optim_1.step()
      self.critic_optim_2.step()
      self.policy_optim.step()

	  #Target network Soft update 
      for target_param, local_param in zip(self.sa_value_target_net_1.parameters(), self.sa_value_net_1.parameters()):
        target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
      for target_param, local_param in zip(self.sa_value_target_net_2.parameters(), self.sa_value_net_2.parameters()):
        target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
      
      return critic_1_loss.detach().item(), critic_2_loss.detach().item(), policy_loss.detach().item(), net_loss, mean_entropy
            
          
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
    return observation, rewards, done_flag, [], []

    # Execute one time step within the environment
    ...
  def reset(self, seed=42):
    self.curr_t = 0
    return np.array([1, 0, 0, 0, 0]), None
    # Reset the state of the environment to an initial state

  def render(self, mode='human', close=False):
    pass

def run(env_name = 'ipd', num_episodes=NUM_EPISODES, reset_interval=RESET_INTERVAL):    
    global EP_LENGTH
    EP_LENGTH = 500 if env_name == 'cartpole' else EP_LENGTH
    env = IPDEnv(ep_length=EP_LENGTH) if env_name == 'ipd' else gym.make('CartPole-v1')
    curr_state, _ = env.reset(seed=SEED)
    agent = SAC(num_states=len(curr_state), num_actions=2)
    for ep_num in range(num_episodes):
        curr_state, _ = env.reset(seed=SEED)
        curr_state = torch.tensor(curr_state).float()
        ep_buffer = []
        episode_reward = 0
        actions_taken = []
        rewards_recd = []
        epsilon = EPS_GREEDY_PROB
        epsilon += max(float(ep_num)*0.2/500.0,0.2)
        # eps_greedy_flags_1 = torch.rand(EP_LENGTH+1)> epsilon
        for i in range(EP_LENGTH+1):
            #Choose and execute action
            # action_t = agent.greedy_policy(curr_state) if not eps_greedy_flags_1[i] else np.random.choice([0,1]).item()
            action_t = agent.greedy_policy(curr_state)
            action_t = [action_t, a2_policy()] if env_name == "ipd" else action_t
            actions_taken.append(action_t)
            next_state, rewards, is_done, _, _ = env.step(action_t)
            
            #Compute and edit reward if needed
            reward_1 = rewards[0] if env_name == 'ipd' else rewards
            episode_reward += reward_1
            if env_name == 'cartpole' and is_done and len(ep_buffer) < 400: reward_1 =-5
            
            #Add to buffer and update next state
            buffer_t = torch.tensor(np.hstack((reward_1, curr_state, next_state, action_t)))
            ep_buffer.append(buffer_t)
            curr_state = torch.tensor(next_state).float()
            
            if is_done: break
        # for i in range(len(ep_buffer)-2, -1, -1):
        #   ep_buffer[i][0] = ep_buffer[i+1][0]*GAMMA + ep_buffer[i][0]
        agent.buffer.extend(ep_buffer)
        if len(agent.buffer) < BUFFER_BATCH_SIZE: 
            print("Filling buffer. Skipping update")
            continue
        c1_loss, c2_loss, pol_loss, net_loss, mean_entropy = agent.update()
        wandb.log({"Episode":ep_num,  "Ep Reward": episode_reward, "Critic Loss 1" : round(c1_loss,2), "Critic Loss 2" : round(c2_loss,2), 
                   "Policy loss" : round(pol_loss,2),  "Net": round(net_loss.item(),2), "Entropy":round(mean_entropy.item(),2)})
        print(f"Episode: {ep_num} \t Ep Reward : {episode_reward} \t Critic Losses : {round(c1_loss,2)} \t | {round(c2_loss,2)}. \t Policy loss : {round(pol_loss,2)} \t Net : {round(net_loss.item(),2)} \t Entropy:{round(mean_entropy.item(),2)}")
        # log results
        if ep_num % args.log_interval == 0:
            defect_probs = {}
            for agent_ind, agent_model in enumerate([agent]):
                for state_ind, state_name in enumerate(MARKOVIAN_STATES):
                    state = torch.zeros((5))
                    state[state_ind] = 1
                    _, state_action_probs, _ = agent_model.sample_policy(state)
                    print("A%d Defect Prob - %s : %r"%(agent_ind, state_name, state_action_probs.detach()[1]))
                    defect_probs["A%d Defect Prob - %s"%(agent_ind, state_name)] = state_action_probs.detach()[1]
            wandb.log(defect_probs)
    
if __name__ == "__main__":
    RUN_ENV = 'ipd'
    exp_args = {'game':RUN_ENV,
                'batch_size':BUFFER_BATCH_SIZE, 
                'buffer_len': BUFFER_LEN, 
                'critic_lr': CRITIC_LR,
                'actor_lr': ACTOR_LR,
                'epsilon': EPS_GREEDY_PROB, 
                'tau':TAU,
                'gamma':GAMMA,
                'alpha':ALPHA,
                'ep_length':EP_LENGTH
                }
    wandb.init(project="IPD Actor-Critic v2", config=exp_args)
    run(env_name=RUN_ENV)