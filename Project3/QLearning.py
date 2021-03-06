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
# from SARSAOP import 

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

class QLearning:

    def __init__(self, alpha, gamma, epsilon):
        # initialize q (sa) for all a's and s's
        # self.qvalue = np.random.random_sample((env.observation_space.n, env.action_space.n))
        self.qvalue = np.zeros((env.observation_space.n, env.action_space.n))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        #print self.qvalue
    def policyevaluation(self):

        R = []
        R_testing = []
        episode_i = 0

        # get the start state
        
        for episode in range(7000):
            # TRAINING
            s = env.reset()
            rewards_i = 0

            for step in range(50):
                count = 0
                # env.render()
                # choose A from the S
                
                action = self.epsilongreedy(s)

                obs2, reward, terminal, _ = env.step(action)
                rewards_i += reward
                self.qvalue[s][action] = self.qvalue[s][action] + self.alpha * \
                (reward + self.gamma * np.max(self.qvalue[obs2]) - self.qvalue[s][action])
                if terminal:
                    count = step
                    # self.qvalue[s] = 0
                    # env.render()
                    break
                s = obs2
                count += 1
                # time.sleep(0.5)
            R.append(rewards_i)  

            # TESTING
        episode_i += 1
        #plt.plot(R)
        # rolling_sum = pandas.rolling_sum(pandas.DataFrame(R_testing), 50)
        # print (rolling_sum)
        plt.plot(R)
        #plt.plot(R_testing)
        print self.qvalue
        print len(R)


    def epsilongreedy(self, s):
        random_num = np.random.random()
        if random_num < self.epsilon:
            return randint(0, (env.action_space.n-1))
        else:
            return np.argmax(self.qvalue[s])

# Main Function
def main():
    
    ql = QLearning(0.1, 0.99, 0.1)
    ql.policyevaluation()
    plt.xlabel("Episodes(Number)", fontsize = 15)
    plt.ylabel(" Total Rewards per episode", fontsize = 15)
    plt.title("Performance Q-Learning TAXI Environment", fontsize = 20)
    plt.grid(True)

    plt.show()  

if __name__ == '__main__':
    main()
