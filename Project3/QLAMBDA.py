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
# env = gym.make('Taxi-v2')
#uncomment for Frozen lake
env = gym.make('FrozenLake-v0')
#env = gym.make('Blackjack-v0')
# env = gym.make('CartPole-v0')
# You have to import MDPGridworldEnv properly in order for environment to work
#env = MDPGridworldEnv()

# prints out that both states and actions are discrete and their valid values
print env.observation_space
print env.action_space

# to access the values
print env.observation_space.n # env.nS
print env.action_space.n # env.nA

# added delay here so you can view output above
time.sleep(2)

class Qlambda:

    def __init__(self, alpha, gamma, epsilon, lambdda):
        # initialize q (sa) for all a's and s's
        self.qvalue = np.random.random_sample((env.observation_space.n, env.action_space.n))
        self.eligibility = np.zeros((env.observation_space.n, env.action_space.n))

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
        
        for episode in range(3000):
            # TRAINING
            s = env.reset()
            rewards_i = 0
            action = self.epsilongreedy(s)

            for step in range(100):
                count = 0
                # env.render()
                s_dash, reward, terminal, _ = env.step(action)
                rewards_i += reward
                a_dash = self.epsilongreedy(s_dash)
                a_star = np.argmax(self.qvalue[s_dash])
                delta = reward + self.gamma * self.qvalue[s_dash][a_star] - self.qvalue[s][action]
                self.eligibility[s][action] += 1
                for s in range(env.observation_space.n):
                    for action in range(env.action_space.n):

                        self.qvalue[s][action] = self.qvalue[s][action] + \
                        self.alpha * delta * self.eligibility[s][action]
                        if (a_dash == a_star):
                            self.eligibility[s][action] = self.gamma * self.lambdda * self.eligibility[s][action]
                        else:
                            self.eligibility[s][action] = 0

                s = s_dash
                action = a_dash

                if terminal:
                    count = step
                    # env.render()
                    break
                count += 1
                # time.sleep(0.5)
            R.append(rewards_i)  

            # TESTING
            # s = env.reset()
            # rewards_i = 0

            # for step in range(50):
            #     count = 0
            #     env.render()
            #     # choose A from the S
                
            #     action = np.argmax(self.qvalue[s]) #TODO

            #     obs2, reward, terminal, _ = env.step(action)
            #     rewards_i += reward
            #     if terminal:
            #         count = step
            #         #self.qvalue[s] = 0
            #         env.render()
            #         break
            #     s = obs2
            #     count += 1

            # R_testing.append(rewards_i)
        episode_i += 1
        #plt.plot(R)
        # rolling_sum = pandas.rolling_sum(pandas.DataFrame(R), 15)
        # print (rolling_sum)
        plt.plot(R)
        # plt.plot(R_testing)
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


    A = [0.9] # List of alphas
    # G = [] # List of Gammas
    L = [0.1] # List of Lambdas
    for i in  L:
        # def __init__(self, alpha, gamma, epsilon, lambdda)
        qlam = Qlambda(0.9, 0.9, 0.1, i)
        qlam.policyevaluation()

    #plt.legend(['lambda = 0.1' , 'lambda = 0.3', \
#         'lambda = 0.5', 'lambda = 0.9'])

    plt.xlabel("Episodes(Number)", fontsize = 15)
    plt.ylabel(" Total Rewards per episode", fontsize = 15)
    plt.title("Frozen Lake(Stochastic)(Watkin's Q(lambda))", fontsize = 20)
    plt.grid(True)

    plt.show()  

if __name__ == '__main__':
    main()
