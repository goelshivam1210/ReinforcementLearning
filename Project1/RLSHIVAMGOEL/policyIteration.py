import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv
import random
from operator import itemgetter
import numpy as np
import gym
from random import randint
import matplotlib.pyplot as plt 
# Uncomment if you added MDPGridworld as a new gym environment
#env = MDPGridworldEnv() 
# Uncomment if you want to use Frozen Lake as a new gym environment
env = gym.make('FrozenLake-v0')
# Uncomment if you want to use Taxi as a new gym environment
#env =  gym.make('Taxi-v2')
env.reset()


# You have to import MDPGridworldEnv properly in order for environment to work
#env = MDPGridworldEnv()

# prints out that both states and actions are discrete and their valid values
print env.observation_space
print env.action_space

# to access the values
print env.observation_space.n # env.nS
print env.action_space.n # env.nA

# added delay here so you can view output above
time.sleep(4)

class PolicyIteration():
    #declared the list for plotting function
    G = [] # Gammas
    R = [] # Rewards
    T = [] # Thetas
    C = [] # Counts(Iterations)

    def initialize(self, g, t):
        self.gamma = g
        self.theta = t
        #inititlalize the values to be 0
        self.value = [0] * env.observation_space.n
        
        #actions for all the states
        self.actions = []
        for i in range(env.action_space.n):
            self.actions.append(i)

        #all the states
        self.states = []
        for i in range(env.observation_space.n):
            self.states.append(i)
        
        #initialize an arbitary policy for each state
        self.policy = []
        #initialize the policy to be random at start
        for i in range(env.observation_space.n):
            self.policy.append(randint(0, env.action_space.n-1))

    def evaluate_policy(self):
        V_s = []
        while True:
            delta = 0
            for state in range(env.observation_space.n):
                V_s_i = 0
                value_old = self.value[state]
                T = env.P[state][self.policy[state]]
                for i in range(len(T)):
                    V_s_i += (T[i][0] * (T[i][2] + self.gamma * self.value[T[i][1]]))
                    # time.sleep(.5)
                self.value[state] = V_s_i
                delta = max(delta, abs((self.value[state] - value_old)))
                # time.sleep(1)
            #calculate the total number of iterations to converge
            self.count += 1
            if delta < self.theta:
                break

        # print self.value
        #print out the optimal set of Values of each state
        # value_mat = np.matrix(self.value)
        # value_mat.shape = (env.observation_space.n/env.action_space.n, env.observation_space.n/env.action_space.n)
        # print ("V(S) : \n {}").format(value_mat)

    def policy_improvement(self):
        policy_stable = True
        for state in range(env.observation_space.n):
            policy_state = []
            #save old policy for that state
            old_action = self.policy[state]
            for action in self.actions:
                policy_state_i = 0
                T = env.P[state][action]
                #print T
                for i in range(len(T)):
                    policy_state_i += T[i][0] * (T[i][2] + self.gamma * self.value[T[i][1]])
                policy_state.append((action,policy_state_i))
            # print max(policy_state, key = itemgetter(1))
            self.policy[state] = max(policy_state, key = itemgetter(1))[0]
            if old_action != self.policy[state]:
                policy_stable = False
        # print self.policy
        # policy_mat = np.matrix(self.policy)
        # policy_mat.shape = (env.observation_space.n/env.action_space.n, env.observation_space.n/env.action_space.n)
        # print ("PI(S) : \n {}").format(policy_mat)
        return policy_stable
        
    def plot(self, x = [], y = []):
        self.xi = []
        self.yi = []
        for item in x:
            self.xi.append(item)
        for item in y:
            self.yi.append(item)
        #plot    
        plt.plot(self.xi, self.yi)
        plt.xlabel("Gamma")
        plt.ylabel("Total Iterations")
        plt.title("Gamma versus Iterations in FROZEN LAKE")
        plt.grid(True)
        plt.show() 
   
def main():
    pi = PolicyIteration()

    G = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    T = [0.001, 0.002, 0.003, 0.004, 0.005]
    file = open('Data2.txt', 'w')
    file.write("Frozen Lake (Stochastic Environment)\n")
    file.write("Gamma      Theta     Averge_Reward      Count\n")
    #iterate to use different values of Theta(T)
    for i in range(len(T)):
        #iterate through the different values of G(gamma)
        for j in range(len(G)):
            pi.initialize(G[j], T[i])
            policy_stable = False
            pi.count = 0
            while policy_stable != True:
                pi.evaluate_policy()
                # time.sleep(1)
                policy_stable = pi.policy_improvement()
            # time.sleep(1)
            obs = env.reset()
            print "Policy Stable---------->>>>>>> {}".format(policy_stable)
            print "Count------------------->>>>>> {}".format(pi.count)
            reward_i = 0
            rewards = 0
            total_rewards = 0
            average_reward = 0
            #once the policy is stable then we can iterate through the environment 100 times to get average rewards
            for u in range(100):
                for t in range(1000):
                    env.render()
                    action = pi.policy[obs]
                    print ("The action is {}".format(action))
                    obs2, reward, terminal, _ = env.step(action)
                    reward_i += reward
                    if terminal:
                        env.render()
                        print("Episode finished after {} timesteps".format(t+1))
                        rewards = reward_i
                        reward_i = 0
                        print ("Total Reward {}".format(rewards))
                        break
                    obs = obs2
                obs = env.reset()    
                total_rewards += rewards
            average_reward = total_rewards/100
            file.write("{}      {}          {}             {}   ".format(G[j], T[i], average_reward, pi.count))
            file.write("\n")

            #for plotting purposes
            pi.G.append(G[j])
            pi.R.append(average_reward)
            pi.T.append(T[i])
            pi.C.append(pi.count)


    pi.plot(pi.G, pi.C)

if __name__ == '__main__':
    main()

    # def policy_improvement(self, ):
