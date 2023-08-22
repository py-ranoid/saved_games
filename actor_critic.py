import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import wandb
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
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--reset-interval', type=int, default=200, metavar='N', help='interval between resets (default: 10)')
parser.add_argument('--buffer-size', type=int, default=512, metavar='N', help='replay buffer size (default: 512)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='number of transitions sampled to update agents (default: 256)')
parser.add_argument('--num-episodes', type=int, default=750, metavar='N', help='number of episodes (default: 750)')
parser.add_argument('--epsilon', type=float, default=0.8, metavar='N', help='epsilon for behaviour policy (default: 0.8)')
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
    observation = np.hstack([0, (actions[0] == 0) & (actions[1] == 0), (actions[0] == 0) & (actions[1] == 1), (actions[0] == 1) & (actions[1] == 0), (actions[0] == 1) & (actions[1] == 1)])
    observation = observation.astype(np.float32)

    return observation, rewards

torch.manual_seed(args.seed)



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
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

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, states_size = STATES_LEN, num_actions=2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(states_size, 128)
        self.states_size = states_size
        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.affine2 = nn.Linear(states_size + num_actions + 1, 128)
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.ep_buffer = []
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=1000))
        
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
        print (action_probs)
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

    def update(self):
        policy_losses = []
        value_losses = []
        buffer_samples = self.buffer.sample(batch_size=BUFFER_BATCH_SIZE)
        wandb.log({"Buffer A1 mean":buffer_samples[:,12].mean()})
        wandb.log({"Buffer A2 mean":buffer_samples[:,13].mean()})

        for sample in buffer_samples:
            # buffer_t_1 = torch.cat((0 - reward1, 1- reward1, 2- curr_state, 7 - next_state, 12 - action1, 13 - action2, 14 - next_action1, 15 - t))
            a_curr, a_next     = sample[12], sample[14]
            curr_state, next_state = sample[2:self.states_size+2], sample[self.states_size+2:2+self.states_size*2]
            ep_t = sample[15+2*self.states_size-10]+1
            q_s_a      = self.get_Q_values(state = curr_state, action=np.hstack([int(a_curr==0), int(a_curr==1)]), t=ep_t)
            
            # Calculate critic loss (TD way)
            q_s_a_dash = self.get_Q_values(state = next_state, action=np.hstack([a_next==0, a_next==1]), t=ep_t)
            value_losses.append(F.huber_loss(q_s_a, GAMMA * q_s_a_dash + sample[0]))
            
            # Log prob of curr_a at curr_state 
            log_prob = torch.log(self.forward(curr_state)[0])[a_curr.int()]
            advantage = q_s_a.detach() - sample[0]
            
            # calculate actor (policy) loss
            policy_losses.append(log_prob * advantage)

        # perform backprop
        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        loss = policy_loss + value_loss
        loss.backward()
        return policy_loss.detach(), value_loss.detach()


def flip_state(curr_state_1):
    if curr_state_1[0] or curr_state_1[1] or curr_state_1[4]: 
        return curr_state_1
    else: return np.array([0,0,1,1,0]) - curr_state_1

def main_cartpole():
    
    env = gym.make('CartPole-v1')
    curr_state, _  = env.reset(seed=args.seed)
    agent1 = Policy(states_size=len(curr_state))
    optimizer1 = optim.Adam(agent1.parameters(), lr=LEARNING_RATE)
    running_reward = 10

    # run infinitely many episodes
    
    for i_episode in count(1):

        # reset environment and episode reward
        agent1.ep_buffer = []
        eps_greedy_flags_1 = torch.rand(EP_LENGTH+1)> EPS_GREEDY_PROB
        curr_state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, EP_LENGTH):

            # select action from policy
            action1 = torch.tensor(np.random.choice([0,1])) if eps_greedy_flags_1[t] else agent1.select_action(curr_state)            

            # take the action
            next_state, reward1, done, _, _ = env.step(action1.numpy())
            reward2 = 0
            next_action1 = agent1.select_action(next_state)
            ep_reward += reward1
            buffer_t_1 = torch.tensor(np.hstack((reward1, reward1, curr_state, next_state, action1, action1, next_action1, t, reward1 + reward2)))
            if args.render:
                env.render()
            curr_state = next_state
            # model.rewards.append(reward)
            agent1.ep_buffer.append(buffer_t_1)
            if done:
                break

        agent1.ep_buffer[-1][0] = ep_reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        for i in range(len(agent1.ep_buffer)-2,-1,-1):
            agent1.ep_buffer[i][0] += GAMMA * agent1.ep_buffer[i+1][0]
        agent1.buffer.extend(agent1.ep_buffer)

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))


        # perform backprop if buffer has sufficient samples
        if EP_LENGTH * (i_episode+1) > BUFFER_BATCH_SIZE:
            # Update agent 1 weights
            optimizer1.zero_grad()
            agent1_loss = agent1.update()
            optimizer1.step()
            wandb.log({"Agent 1 Loss": agent1_loss})
        else:
            print("Filling buffer")
            continue

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

            # for agent_ind, agent_model in enumerate([agent1]):
            #     for state_ind, state_name in enumerate(MARKOVIAN_STATES):
            #         state = torch.zeros((5))
            #         state[state_ind] = 1
            #         state_action_probs, _ = agent_model(state)
            #         wandb.log({"A%d Defect Prob - %s"%(agent_ind, state_name) :state_action_probs.detach()[1]})
                    
        if i_episode % RESET_INTERVAL == 0:
            for agent in [agent1]:
                torch.nn.init.xavier_uniform(agent.affine1.weight)
                torch.nn.init.xavier_uniform(agent.action_head.weight)
                torch.nn.init.xavier_uniform(agent.affine2.weight)
                torch.nn.init.xavier_uniform(agent.value_head.weight)
            wandb.log({"Reset": 1})
            print ("--------RESET--------")
        else:
            wandb.log({"Reset": 0})                

def main():    
    agent1 = Policy()
    agent2 = Policy()
    optimizer1 = optim.Adam(agent1.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.Adam(agent2.parameters(), lr=LEARNING_RATE)

    for i_episode in range(NUM_EPISODES):

        # reset environment and episode reward
        curr_state = np.hstack([1, 0, 0, 0, 0])
        eps_greedy_flags_1 = torch.rand(EP_LENGTH)> EPS_GREEDY_PROB
        eps_greedy_flags_2 = torch.rand(EP_LENGTH)> EPS_GREEDY_PROB
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        agent1.ep_buffer = []
        agent2.ep_buffer = []
        for t in range(EP_LENGTH):

            # select action from policy
            action1 = torch.tensor(np.random.choice([0,1])) if eps_greedy_flags_1[t] else agent1.select_action(curr_state)
            action2 = torch.tensor(np.random.choice([0,1])) if eps_greedy_flags_2[t] else agent2.select_action(flip_state(curr_state)) 
            action = np.hstack((action1, action2))
            # take the action
            # state, reward, done, _, _ = env.step(action)
            next_state, rewards = ipd_step([action1, action2])
            reward1 = [rewards[0]]
            reward2 = [rewards[1]]
            # TODO: Change state perspective
            next_action1 = agent1.select_action(next_state)
            next_action2 = agent2.select_action(flip_state(next_state))
            buffer_t_1 = torch.tensor(np.hstack((reward1, reward1, curr_state, next_state, action1, action2, next_action1, t, reward1 + reward2)))
            buffer_t_2 = torch.tensor(np.hstack((reward2, reward2, curr_state, next_state, action2, action1, next_action2, t, reward1 + reward2)))
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
            # Update agent 1 weights
            optimizer1.zero_grad()
            agent1_loss = agent1.update()
            optimizer1.step()
            # Update agent 2 weights
            optimizer2.zero_grad()
            agent2_loss = agent2.update()
            optimizer2.step()
            wandb.log({"Agent 1 Loss": agent1_loss, "Agent 2 Loss": agent2_loss})
        else:
            print("Filling buffer")
            continue

        # log results
        if i_episode % args.log_interval == 0:
            print("Episode : ", i_episode)            
            for agent_ind, agent_model in enumerate([agent1, agent2]):
                for state_ind, state_name in enumerate(MARKOVIAN_STATES):
                    state = torch.zeros((5))
                    state[state_ind] = 1
                    state_action_probs, _ = agent_model(state)
                    wandb.log({"A%d Defect Prob - %s"%(agent_ind, state_name) :state_action_probs.detach()[1]})
                    
        if i_episode % RESET_INTERVAL == 0:
            for agent in [agent1, agent2]:
                torch.nn.init.xavier_uniform(agent.affine1.weight)
                torch.nn.init.xavier_uniform(agent.action_head.weight)
                torch.nn.init.xavier_uniform(agent.affine2.weight)
                torch.nn.init.xavier_uniform(agent.value_head.weight)
            wandb.log({"Reset": 1})
            print ("--------RESET--------")
        else:
            wandb.log({"Reset": 0})
            

if __name__ == '__main__':
    exp_args = {'game':'cartpole','reset_interval':RESET_INTERVAL, 'batch_size':BUFFER_BATCH_SIZE, 'buffer_len': BUFFER_LEN, 'learning_rate':LEARNING_RATE, 'epsilon': EPS_GREEDY_PROB}
    print(exp_args)
    wandb.init(project="IPD Actor-Critic", config=exp_args)
    main_cartpole()
