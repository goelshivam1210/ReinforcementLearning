# On Policy First Visit MC
import gym # Uncomment if using gym.make
import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv
import random
from operator import itemgetter
import numpy as np
from random import randint
import matplotlib.pyplot as plt 
import pandas 

# Uncomment if you added MDPGridworld as a new gym environment
#uncomment for Frozen lake
env = gym.make('Taxi-v2')
#uncomment for Frozen lake
#env = gym.make('FrozenLake-v0')
#env = gym.make('Blackjack-v0')
# You have to import MDPGridworldEnv properly in order for environment to work
# env = MDPGridworldEnv()

# prints out that both states and actions are discrete and their valid values
print env.observation_space
print env.action_space

# to access the values
print env.observation_space.n # env.nS
print env.action_space.n # env.nA

# added delay here so you can view output above
time.sleep(2)

class Sarsa:

    def __init__(self, alpha, gamma, lambdda, epsilon):
        # initialize q (sa) for all a's and s's
        self.qvalue = np.random.random_sample((env.observation_space.n, env.action_space.n))
        # self.qvalue = np.zeros((env.observation_space.n, env.action_space.n))
        # self.eligibility = np.zeros((env.observation_space.n, env.action_space.n))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambdda = lambdda
        #print self.qvalue
    def policyevaluation(self):

        R = []
        R_testing = []
        episode_i = 0

        # get the start state
        
        for episode in range(10000):
            # TRAINING
            s = env.reset()
            action = self.epsilongreedy(s)
            rewards_i = 0

            for step in range(50):
                count = 0
                #env.render()
                # choose A from the S
                
                s_dash, reward, terminal, _ = env.step(action)
                a_dash = self.epsilongreedy(s_dash)
                self.qvalue[s][action] = self.qvalue[s][action] + self.alpha * \
                (reward + self.gamma * (self.qvalue[s_dash][a_dash]) - self.qvalue[s][action])
                s = s_dash
                a = a_dash

                rewards_i += reward
                if terminal:
                    count = step
                    # self.qvalue[s] = 0
                    env.render()
                    break
                count += 1
                # time.sleep(0.5)
            R.append(rewards_i) 
            #print R
            #plt.plot(R)
        # rolling_sum = pandas.rolling_sum(pandas.DataFrame(R), 25)
        # print (rolling_sum)
        plt.plot(R)
        print self.qvalue
        #plt.plot(R_testing)
        # print self.qvalue

    def epsilongreedy(self, s):
        random_num = np.random.random()
        if random_num < self.epsilon:
            return randint(0, (env.action_space.n-1))
        else:
            return np.argmax(self.qvalue[s])

# Main Function
def main():
    sar = Sarsa(0.1, 0.99,0.9, 0.1)
    sar.policyevaluation()
    plt.grid(True)
    plt.xlabel("Episodes(Number)", fontsize = 15)
    plt.ylabel(" Total Rewards per episode", fontsize = 15)
    plt.title("Performance of SARSA in TAXI Environment")
    plt.show()  

if __name__ == '__main__':
    main()
