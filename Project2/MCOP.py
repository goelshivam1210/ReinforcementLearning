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
#env = gym.make('Taxi-v2')
#uncomment for Frozen lake
#env = gym.make('FrozenLake-v0')
#env = gym.make('Blackjack-v0')
# You have to import MDPGridworldEnv properly in order for environment to work
env = MDPGridworldEnv()

# prints out that both states and actions are discrete and their valid values
print env.observation_space
print env.action_space

# to access the values
print env.observation_space.n # env.nS
print env.action_space.n # env.nA

# added delay here so you can view output above
time.sleep(2)

class MCOP:

    # Function to initialize the values

    def initialize(self, eps):

        # Q values
        self.q = np.zeros((env.observation_space.n, env.action_space.n), dtype = 'float')
        # Returns Sum List 
        self.returns_sum = np.zeros((env.observation_space.n, env.action_space.n), dtype = 'float')
        #Returns Count List Since we average out the total return so number is required
        self.returns_count = np.zeros((env.observation_space.n, env.action_space.n), dtype = 'float')
        # Probabilities of the state action pairs
        self.pi = np.tile(1/float(env.action_space.n), (env.observation_space.n, env.action_space.n))
        # Value for randomness
        self.epsilon = eps

    # Evaluating the stable policy during exploration
    def evaluation(self):
        #list for storing Rewards
        R = []
        episode_i = 0
        # No of episodes to be generated, used 10000 for Frozen Lake, 3000 for taxi and 200 for MDP
        for i in range(200):
            pi_old = np.copy(self.pi)
            #print ("Old Policy \n {}").format(pi_old)
            count = 0
            E = []
            obs = env.reset()
            reward_i = 0
            for t in range(100):
                #env.render()
                action = np.random.choice(env.action_space.n , p = self.pi[obs])
                #print ("Action {}").format(action)
                obs2, reward, terminal, _ = env.step(action)
                E.append((obs, action, reward))
                reward_i += reward
                if terminal:
                    count = t
                    break
                #update the new state    
                obs = obs2
            # print ("Episodes {}".format(E))
            # time.sleep(4)
                count += 1
        
            # to iterate over the list E to find the first occurence of the state in the list of episodes
            for s, a, r in E:
                # CalculateG returns the first occurence of the state
                G = self.calculateG(self.findFirstOccurrence(s, a, E), E)
                #calculate the sum
                self.returns_sum[s][a] += G
                #calculate the count to average out the value
                self.returns_count[s][a] += 1
                #update the Q values of the state 
                self.q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]


            for s, _, _ in E:
                A = np.argmax(self.q[s])
                # iterate over the actions space to find the best action for the given state and update the value
                # of the probability function according to the formula
                for a in range(env.action_space.n):
                    if A == a:
                        self.pi[s][a] = 1 - self.epsilon + (self.epsilon/env.action_space.n)
                    else:
                        self.pi[s][a] = self.epsilon/env.action_space.n
            episode_i += 1

            # print ("New Policy \n {}").format(self.pi)
            # print count
            # print reward_i/count
            # print episode_i
            if not np.array_equal(pi_old, self.pi):
                for s in range(env.observation_space.n):
                    #print np.argmax(self.pi[s])
                #     file.write("{} ".format(np.argmax(self.pi[s])))

                # file.write("\n \n \n")

            # for s,a in pi_old:
            #     cntr = 0
            #     if pi_old[s][a] != self.pi[s][a]:
            #         cntr += 1
            #     if cntr > 0 :
            #         print ("Policy Changes------->>>>")
            #         for s in self.pi:
            #             print np.argmax(self.pi[s])

            #R.append(reward_i/count)
            R.append(reward_i)

        #print ("Rewards------>>>>>>{}").format(R)
        #print R

        # calculate the rolling window average of the rewards by choosing the window size
        rolling_sum = pandas.rolling_sum(pandas.DataFrame(R), 1)
        plt.plot(rolling_sum)

    # Function to return of the index of the first occurence of the state in the Episode
    def findFirstOccurrence(self, s, a, E):
        for idx, exp in enumerate(E):
            if exp[0] == s and exp[1] == a:
                return idx

    # Function to calculate the times it occurs and the value it gets
    def calculateG (self, idx, E):
        G = 0
        for _, _, r in  E[idx:]:
            G += r
        return G

# Main Function
def main():

    mc = MCOP()
    # various values of the epsilon
    Eps = [0.1, 0.5, 0.05]
    # iterate over all the epsilons to generate different graphs
    for i in Eps:
        mc.initialize(i)
        mc.evaluation()

    # plotting schema    
    plt.xlabel("Episodes", fontsize = 15)
    plt.ylabel("Average Rewards", fontsize = 15)
    plt.title("Comparison of the Epsilons in MDP Environment", fontsize = 20)
    plt.grid(True)
    plt.legend(['epsilon = 0.1','epsilon = 0.5','epsilon = 0.05'], loc = 4)

    plt.show()


if __name__ == '__main__':
    main()
