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
from pprint import pprint

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
	DEVICE = torch.device('cuda')
 
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--render', action='store_true')
parser.add_argument('--log-interval', type=int, default=5)
parser.add_argument('--reset-interval', type=int, default=10000)
parser.add_argument('--reset-offset', type=int, default=-1)
parser.add_argument('--buffer-size', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--agent2', type=str, default='tft')
parser.add_argument('--num-episodes', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--alr', type=float, default=0.003)
parser.add_argument('--clr', type=float, default=0.003)
parser.add_argument('--shared', type=int, default=0)
args = parser.parse_args()

POL_HID_SIZE = 128
MARKOVIAN_STATES = ['XX', 'CC', 'CD', 'DC', 'DD'] 
STATES_LEN = len(MARKOVIAN_STATES)
EP_LENGTH = 50
RESET_INTERVAL = args.reset_interval
RESET_OFFSET = int(args.reset_interval / 2) if args.reset_offset == -1 else args.reset_offset
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
						nn.Linear(hidden_size, out_size)).to(DEVICE)

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
		self.is_learning = True
		self.agent_num = 0  
		
		# Initialize policy net
		self.policy_net     = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
		self.policy_net.append(nn.Softmax(dim=-1))
		self.policy_net.apply(reset_weights)

		# Initialize Q nets and copy weights to target
		self.sa_value_net_1         = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
		self.sa_value_net_2         = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
		self.sa_value_target_net_1  = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
		self.sa_value_target_net_2  = create_net(in_size=num_states, hidden_size=128, out_size= num_actions)
		self.sa_value_net_1.apply(reset_weights)
		self.sa_value_net_2.apply(reset_weights)
		self.sa_value_target_net_1.load_state_dict(self.sa_value_net_1.state_dict())
		self.sa_value_target_net_2.load_state_dict(self.sa_value_net_2.state_dict())        

		# Initialize alpha
		# self.log_alpha = torch.zeros(1, requires_grad=False, device=DEVICE)
		self.alpha = torch.ones(1, requires_grad=True, device=DEVICE)
		#TODO: Skipping alpha optimisation

		#Intialise replay buffer
		self.buffer = ReplayBuffer(storage=ListStorage(max_size=BUFFER_LEN))
		
		#Initialise optimisers
		self.critic_optim_1 = torch.optim.Adam(self.sa_value_net_1.parameters(), lr=CRITIC_LR, eps=1e-4)
		self.critic_optim_2 = torch.optim.Adam(self.sa_value_net_2.parameters(), lr=CRITIC_LR, eps=1e-4)
		self.policy_optim   = torch.optim.Adam(self.policy_net.parameters(), lr=ACTOR_LR, eps=1e-4)
		self.alpha_optim    = torch.optim.Adam([self.alpha], lr=ACTOR_LR/10, eps=1e-5)
		self.alpha = self.alpha * ALPHA

		
	def forward(self):
		pass
		
	def sample_policy(self, states):
		action_probs = self.policy_net(states.to(DEVICE))
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

	def _reset_grads(self):
		self.critic_optim_1.zero_grad()
		self.critic_optim_2.zero_grad()
		self.policy_optim.zero_grad()
		self.alpha_optim.zero_grad()
	
	def _run_optim(self):
		self.critic_optim_1.step()
		self.critic_optim_2.step()
		self.policy_optim.step()
		self.alpha_optim.step()
	
	def update(self, wandb_dict={}):
		#Sample transitions from buffer
		buffer_samples = self.buffer.sample(batch_size=BUFFER_BATCH_SIZE)
		policy_losses = []
		policy_entropies = []
		policy_q_losses = []
		alpha_losses = []
		entropies = []
		###################################################################################################
  		###################################################################################################

		#Reset critic optim gradients 
		critic_1_losses = []
		self.critic_optim_1.zero_grad()

		# Computing Actor loss by minimizing TD Error wrt target
		for sample in buffer_samples:
			reward, curr_state, next_state, buff_action = self.unpack_buffer(sample)
			curr_state, next_state = curr_state.to(DEVICE), next_state.to(DEVICE)

			q1_s_a = self.sa_value_net_1(curr_state)[buff_action]
			q_s_a_target = reward + GAMMA * self.calc_state_val(next_state)
			critic_1_losses.append(F.huber_loss(q1_s_a, q_s_a_target.detach()))
			  
		# Backpropogate critic losses
		critic_1_loss = torch.stack(critic_1_losses).sum()
		critic_1_loss.backward()
		self.critic_optim_1.step()
		##################################################
		
  		#Reset critic optim gradients 
		critic_2_losses = []
		self.critic_optim_2.zero_grad()	  

		# Computing Actor loss by minimizing TD Error wrt target
		for sample in buffer_samples:
			reward, curr_state, next_state, buff_action = self.unpack_buffer(sample)
			curr_state, next_state = curr_state.to(DEVICE), next_state.to(DEVICE)

			q2_s_a = self.sa_value_net_2(curr_state)[buff_action]
			q_s_a_target = reward + GAMMA * self.calc_state_val(next_state)
			critic_2_losses.append(F.huber_loss(q2_s_a, q_s_a_target.detach()))
			  
		# Backpropogate critic losses
		critic_2_loss = torch.stack(critic_2_losses).sum()
		critic_2_loss.backward()
		self.critic_optim_2.step()
		###################################################################################################
  		###################################################################################################

  		#Reset policy optim gradients 
		self.policy_optim.zero_grad()
		# Computing policy net losses by Eq 9
		for sample in buffer_samples:
			_ , curr_state, _ , _ = self.unpack_buffer(sample)
			curr_state = curr_state.to(DEVICE)

			#Minimizing entropy loss and maximising state values of actions with high prob
			min_s_val = torch.min(self.sa_value_net_1(curr_state), self.sa_value_net_2(curr_state))
			_, action_probs, log_probs = self.sample_policy(curr_state)
			policy_entropy    = (action_probs * log_probs).sum(dim=-1)
			policy_q_loss     = (- action_probs * min_s_val.detach()).sum(dim=-1)        
			policy_loss       = self.alpha.detach() * policy_entropy + policy_q_loss

			policy_losses.append(policy_loss)
			policy_entropies.append(policy_entropy)
			policy_q_losses.append(policy_q_loss)

			# Computing alpha loss by Eq 11
			# target_ent = ALPHA
			# alpha_losses.append(action_probs.detach() * (-self.alpha * (log_probs + target_ent).detach()))
			# alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        
		net_policy_loss   	 = torch.stack(policy_losses).sum()
		net_policy_entropy   = torch.stack(policy_entropies).sum()
		net_policy_q_loss    = torch.stack(policy_q_losses).sum()
		# alpha_loss    		 = torch.stack(alpha_losses).sum()
		net_policy_loss.backward()
		self.policy_optim.step()
		#alpha_loss.backward()
		#Target network Soft update 
		for target_param, local_param in zip(self.sa_value_target_net_1.parameters(), self.sa_value_net_1.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
		for target_param, local_param in zip(self.sa_value_target_net_2.parameters(), self.sa_value_net_2.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
		###################################################################################################
  		###################################################################################################

		if "Episode" in wandb_dict:
			wandb_dict.update({	
                      	f"A{self.agent_num} Critic Loss 1" : round(critic_1_loss.detach().item(),2), 
						f"A{self.agent_num} Critic Loss 2" : round(critic_2_loss.detach().item(),2), 
						f"A{self.agent_num} Policy loss"   : round(net_policy_loss.detach().item(),2), 
						f"A{self.agent_num} Poli Q loss"   : round(net_policy_q_loss.detach().item(),2), 
						f"A{self.agent_num} Poli Ent loss" : round(net_policy_entropy.detach().item(),2), 
						f"A{self.agent_num} Net Loss"      : round(((critic_1_loss + critic_2_loss)/2+net_policy_loss).item(),2), 
						f"A{self.agent_num} Alpha"         : self.alpha
						# f"A{self.agent_num} Alpha Loss"    : alpha_loss.detach().item()
      				})
			wandb.log(wandb_dict)

		return wandb_dict

	def reset_nets(self):
		self.policy_net.apply(reset_weights)
		self.sa_value_net_1.apply(reset_weights)
		self.sa_value_net_2.apply(reset_weights)
		self.sa_value_target_net_1.apply(reset_weights)
		self.sa_value_target_net_2.apply(reset_weights)
		
			
class Opponent(SAC):
	
	def __init__(self, num_states, num_actions, response='tft'):
		super(Opponent, self).__init__(num_states, num_actions)
		self.response    = response
		self.is_learning = self.response == 'sac'
		self.agent_num   = 1

	def flip_state(self, curr_state_1):
		# Flips a state wrt 1 
		if curr_state_1[0] or curr_state_1[1] or curr_state_1[4]: 
			return curr_state_1
		else: return torch.tensor([0,0,1,1,0]) - curr_state_1

	def greedy_policy(self, curr_state):
		action_index = int(torch.sum(curr_state * torch.arange(5)))
		if   self.response == 'cop':   # Always Cooperate
			return 0
		elif self.response == 'def': # Always Defect
			return 1
		elif self.response == 'tft': # Tit-for-Tat
			return int(action_index in {2,4})
		elif self.response == 'rnd':
			return int(torch.rand(1)>0.5)
		elif self.response == 'sac': # SAC
			return super().greedy_policy(curr_state)
		
class IPDEnv(gym.Env):
	"""Custom Gym Environment to play IPD"""

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

	def reset(self, seed=42):
		self.curr_t = 0
		return np.array([1, 0, 0, 0, 0]), None
		# Reset the state of the environment to an initial state

	def render(self, mode='human', close=False):
		pass

def run(env_name = 'ipd', num_episodes=NUM_EPISODES, reset_interval=RESET_INTERVAL):
	global EP_LENGTH
	EP_LENGTH = 500 if env_name == 'cartpole' else EP_LENGTH
	
	#Create agent, opponent and environment
	env = IPDEnv(ep_length=EP_LENGTH) if env_name == 'ipd' else gym.make('CartPole-v1')
	curr_state, _ = env.reset(seed=SEED)
	agent =      SAC(num_states=len(curr_state), num_actions=2).to(DEVICE)
	oppnt = Opponent(num_states=len(curr_state), num_actions=2, response=args.agent2).to(DEVICE)
	
	for ep_num in range(num_episodes):
		# Initialise environment, buffer and episode rewards
		curr_state, _ = env.reset(seed=SEED)
		curr_state = torch.tensor(curr_state).float()
		ep_buffer, ep_buffer_2    = [], []
		episode_reward_1, episode_reward_2 = 0, 0

		for i in range(EP_LENGTH+1):
			#Choose and execute action
			action1_t = agent.greedy_policy(curr_state)
			action2_t = oppnt.greedy_policy(oppnt.flip_state(curr_state))
			action_t = [action1_t, action2_t] if env_name == "ipd" else action1_t
			next_state, rewards, is_done, _, _ = env.step(action_t)
			
			#Compute rewards
			if env_name == 'ipd':
				reward_1 = rewards[0] if not args.shared else rewards[0] + rewards[1]
				reward_2 = rewards[1] if not args.shared else rewards[0] + rewards[1]
				episode_reward_2 += reward_2
			else: 
				reward_1 = rewards
			episode_reward_1 += reward_1
			if env_name == 'cartpole' and is_done and len(ep_buffer) < 400: reward_1 =-5
			
			#Add to agent's buffer 
			buffer_t  = torch.tensor(np.hstack((reward_1, curr_state, next_state, action_t)))
			ep_buffer.append(buffer_t)
			
			#Add to opponent's buffer 
			if oppnt.is_learning:
				buffer2_t = torch.tensor(np.hstack((reward_2, oppnt.flip_state(curr_state), oppnt.flip_state(next_state), [action2_t, action1_t])))
				ep_buffer_2.append(buffer2_t)
			
			curr_state = torch.tensor(next_state).float()
			
			if is_done: break

		# Add episode to replay buffers 
		agent.buffer.extend(ep_buffer)
		if oppnt.is_learning:
			oppnt.buffer.extend(ep_buffer_2)
					
		# Train agent and opponent
		if len(agent.buffer) < BUFFER_BATCH_SIZE or (oppnt.is_learning and len(oppnt.buffer) < BUFFER_BATCH_SIZE):
			print("Filling buffer. Skipping update")
			continue
		agent.train()
		wandb_dict_A1 = agent.update(wandb_dict={"Episode":ep_num,  "Ep Reward": episode_reward_1})
		pprint(wandb_dict_A1)
		
		if oppnt.is_learning:
			oppnt.train()
			wandb_dict_A2 = oppnt.update(wandb_dict={"Episode":ep_num,  "Ep Reward": episode_reward_2})
			print("A2 update")
			print(oppnt.agent_num)
			pprint(wandb_dict_A2)
		
		# Log defect probabilities for each state
		if ep_num % args.log_interval == 0:
			agent.eval()            
			oppnt.eval()            
			with torch.no_grad():
				defect_probs = {"Episode":ep_num}
				for agent_model in [agent, oppnt]:
					if agent_model.is_learning:
						for state_ind, state_name in enumerate(MARKOVIAN_STATES):
							state = torch.zeros((5))
							state[state_ind] = 1
							_, state_action_probs, _ = agent_model.sample_policy(state)
							# print("A%d Defect Prob - %s : %r"%(agent_model.agent_num, state_name, state_action_probs.detach()[1]))
							defect_probs["A%d Defect Prob - %s"%(agent_model.agent_num, state_name)] = state_action_probs.detach()[1]
							defect_probs["A%d Defect Val - %s"%(agent_model.agent_num, state_name)]  = agent_model.sa_value_net_1(state.to(DEVICE)).detach()[1]
				wandb.log(defect_probs)
				pprint(defect_probs)
			
		if ep_num % RESET_INTERVAL == 0:
			agent.reset_nets()
			print ("RESET 1")

		if ep_num % RESET_INTERVAL == RESET_OFFSET:
			oppnt.reset_nets()
			print ("RESET 2")
	
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
				'agent2':args.agent2,
				'ep_length':EP_LENGTH,
				'shared':args.shared,
				}
	print(exp_args)
	# for state_ind, state_name in enumerate(MARKOVIAN_STATES):
	#     state = torch.zeros((5))
	#     state[state_ind] = 1
	#     a2_response = a2_policy(state)
	#     print("A%d Defect Prob - %s : %r"%(1, state_name, float(a2_response==1)*100))
	shar = "Shared" if args.shared else "Greedy"
	run_name = f"A2-{args.agent2}-{shar}-RES{RESET_INTERVAL}-RO{RESET_OFFSET}-LR{args.alr}"
	wandb.init(project="IPD Actor-Critic v2", config=exp_args, name=run_name)
	wandb.define_metric("Episode")
	run(env_name=RUN_ENV)