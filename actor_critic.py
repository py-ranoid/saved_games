import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer, ListStorage

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

def ipd_step(actions):
    """
    actions: 0=cooperate, 1:defect, shape: [2x1] = [first player action, second player action], dtype=int
    """
    state = actions[0] * 2 + actions[1]  # CC=0, CD=1, DC=2, DD=3

    # calculate rewards
    player1_rewards = np.array([-1., -3., 0, -2.])
    player2_rewards = np.array([-1., 0., -3, -2.])
    r1 = player1_rewards[state]
    r2 = player2_rewards[state]
    rewards = np.hstack([r1, r2])

    # calculate observation shape: [4x1] = [player1_cooperated, player1_defected, player2_cooperated, player2_defected]
    observation = np.hstack([0, actions[0] == 0, actions[0] == 1, actions[1] == 0, actions[1] == 1])
    observation = observation.astype(np.float32)

    return observation, rewards

env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
MARKOVIAN_STATES = ['--', 'CC', 'CD', 'DC', 'DD'] 
STATES_LEN = len(MARKOVIAN_STATES)
EP_LENGTH = 50
RESET_INTERVAL = 50
BUFFER_BATCH_SIZE = 256
BUFFER_LEN = 1000

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, states_size = STATES_LEN, num_actions=2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(states_size, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.affine2 = nn.Linear(states_size + num_actions + 1, 128)
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.ep_buffer = []
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=1000), size=1000)
        
    def forward(self, x, a=torch.tensor([0,0]), t=torch.tensor([0])):
        """
        forward of both actor and critic
        """
        x1 = F.relu(self.affine1(x.float()))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x1.float()), dim=-1)

        # critic: evaluates being in the state s_t
        x2 = F.relu(self.affine2(torch.cat((x, a, t)).float()))
        state_values = self.value_head(x2)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs, _ = self.forward(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(action_probs)
        action = m.sample()

        # the action to take
        return action.detach()
    
    def get_Q_values(self, state, action, t):
        # state = torch.from_numpy(state).float()
        action = torch.tensor(action)
        t = torch.tensor([t])
        _, state_values = self.forward(state, action, t)
        return state_values


agent1 = Policy()
agent2 = Policy()
optimizer1 = optim.Adam(agent1.parameters(), lr=3e-2)
optimizer2 = optim.Adam(agent2.parameters(), lr=3e-2)
# eps = np.finfo(np.float32).eps.detach()
GAMMA = 0.9

def update(agent_model):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    policy_losses = []
    value_losses = []
    buffer_samples = agent_model.buffer.sample(batch_size=BUFFER_BATCH_SIZE)
    print(buffer_samples.shape)
    for sample in buffer_samples:
        # buffer_t_1 = torch.cat((0 - reward1, 1- reward1, 2- curr_state, 7 - next_state, 12 - action1, 13 - action2, 14 - next_action1, 15 - t))
        a_curr     = sample[12]
        a_next     = sample[14]
        q_s_a      = agent_model.get_Q_values(state = sample[2:7], action=np.hstack([a_curr==0, a_curr==1]), t=sample[15])
        
        # Calculate critic loss (TD way)
        q_s_a_dash = agent_model.get_Q_values(state = sample[7:12], action=np.hstack([a_next==0, a_next==1]), t=sample[15]+1)
        value_losses.append(F.huber_loss(q_s_a, GAMMA * q_s_a_dash + sample[1]))
        
        # Log prob of curr_a at curr_state 
        log_prob = torch.log(agent_model(sample[2:7])[0])[a_curr.int()]
        advantage = sample[0] - q_s_a.detach()
        
        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

    print ("Backpropping losses")
    # reset gradients and sum losses
    optimizer1.zero_grad()
    optimizer2.zero_grad()    
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer1.step()
    optimizer2.step()

def flip_state(curr_state_1):
    if curr_state_1[0] or curr_state_1[1] or curr_state_1[4]: 
        return curr_state_1
    else: return np.array([0,0,1,1,0]) - curr_state_1

def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(100):

        # reset environment and episode reward
        curr_state = np.hstack([1, 0, 0, 0, 0])
        
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        agent1.ep_buffer = []
        agent2.ep_buffer = []
        for t in range(EP_LENGTH):

            # select action from policy
            action1 = agent1.select_action(curr_state)
            action2 = agent2.select_action(flip_state(curr_state))
            action = np.hstack((action1, action2))
            # take the action
            # state, reward, done, _, _ = env.step(action)
            next_state, rewards = ipd_step([action1, action2])
            reward1 = [rewards[0]]
            reward2 = [rewards[1]]
            # TODO: Change state perspective
            next_action1 = agent1.select_action(next_state)
            next_action2 = agent2.select_action(flip_state(next_state))
            buffer_t_1 = torch.tensor(np.hstack((reward1, reward1, curr_state, next_state, action1, action2, next_action1, t)))
            buffer_t_2 = torch.tensor(np.hstack((reward2, reward2, curr_state, next_state, action2, action1, next_action2, t)))
            # agent1.ep_buffer.append({"state": curr_state, "next_state":next_state, "action": action1, "next_action": next_action1, "reward": reward1, "op_action" : action2, "time":t, "return": reward1})
            # agent2.ep_buffer.append({"state": curr_state, "next_state":next_state, "action": action2, "next_action": next_action2, "reward": reward2, "op_action" : action1, "time":t, "return": reward2})
            agent1.ep_buffer.append(buffer_t_1)
            agent2.ep_buffer.append(buffer_t_2)
            curr_state = next_state

            # model.rewards.append(reward)

        # updat cumulative discounted returns and add to buffer
        for i in range(len(agent1.ep_buffer)-2,-1,-1):
            # agent1.ep_buffer[i]['return'] += GAMMA * agent1.ep_buffer[i+1]['return']
            # agent2.ep_buffer[i]['return'] += GAMMA * agent2.ep_buffer[i+1]['return']
            agent1.ep_buffer[i][0] += GAMMA * agent1.ep_buffer[i+1][0]
            agent2.ep_buffer[i][0] += GAMMA * agent2.ep_buffer[i+1][0]
        agent1.buffer.extend(agent1.ep_buffer)
        agent2.buffer.extend(agent2.ep_buffer)

        # perform backprop if buffer has sufficient samples
        if EP_LENGTH * (i_episode+1) > BUFFER_BATCH_SIZE:
            update(agent1)
            update(agent2)

        # log results
        if i_episode % args.log_interval == 0:
            print("Episode : ", i_episode)            
            for agent_ind, agent_model in enumerate([agent1, agent2]):
                print("Action probs for Agent %d"%agent_ind)
                for state_ind, state_name in enumerate(MARKOVIAN_STATES):
                    state = torch.zeros((1,5))
                    state[state_ind] = 1
                    print(state)
                    state_action_probs, _ = agent_model(state)
                    print ("Action probs for %s : %r" % (state_name, state_action_probs))
                    
        if i_episode % RESET_INTERVAL:
            for agent in [agent1, agent2]:
                torch.nn.init.xavier_uniform(agent.affine1.weight)
                torch.nn.init.xavier_uniform(agent.action_head.weight)
                torch.nn.init.xavier_uniform(agent.affine2.weight)
                torch.nn.init.xavier_uniform(agent.value_head.weight)
            

if __name__ == '__main__':
    main()
