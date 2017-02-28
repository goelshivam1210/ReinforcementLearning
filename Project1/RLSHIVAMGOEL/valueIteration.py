#Value Iteration

# import gym # Uncomment if using gym.make
import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv
import random
from operator import itemgetter
import numpy as np

# Uncomment if you added MDPGridworld as a new gym environment
# env = gym.make('MDPGridworld-v0')
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

# for i_episode in range(20):
#     obs = env.reset()
#     for t in range(100):
#         env.render()
#         # time.sleep(.5) # uncomment to slow down the simulation
#         action = env.action_space.sample() # act randomly
#         print ("The action is {}".format(action))
#         obs2, reward, terminal, _ = env.step(action)
#         if terminal:
#             env.render()
#             print("Episode finished after {} timesteps".format(t+1))
#             break

class ValueIteration():

    def initialize(self):
        
        self.value = [0] * env.observation_space.n
        
        #actions for all the states
        self.actions = []
        for i in range(env.action_space.n):
            self.actions.append(i)
            #policy is chosing randomly
            #self.policy = random.choice(self.actions)

        #all the states
        self.states = []
        for i in range(env.observation_space.n):
            self.states.append(i)
        
        self.policy = []
        # for i  in range(env.observation_space.n):
        #     self.policy_state.append(i)

    def evaluate_policy(self):
        V_s = []
        theta = 0.001
        gamma = 0.99
        while True:

            delta = 0

            for state in range(env.observation_space.n):
                value_old = self.value[state] 
                for action in self.actions:
                    state_action = env.P[state][action]
                    #print ("state action results of the desired {}".format(state_action))
                    V_s_i = state_action[0][0] * (state_action[0][2] + gamma * self.value[state_action[0][1]])
                    #print ("State {} Action {} V(S): {}".format(state, action, V_s_i ))
                    V_s.append(V_s_i)
                # print ("V(S) is max() {}".format(V_s))
                self.value[state] = max(V_s)
                V_s = []
                #new_value  += Probabilityofnewstate * (rewardatnewstate + gamma * valueofnewstate)
                delta = max(delta, abs((self.value[state] - value_old)))

            if delta < theta:
                break

        #print self.value
        value_mat = np.matrix(self.value)
        value_mat.shape = (3, 4)
        print ("V(S) : \n {}").format(value_mat)

    def policy_improvement(self):
        gamma = 1
        for state in range(env.observation_space.n):
            policy_state = []
            for action in self.actions:
                state_action = env.P[state][action]
                policy_state_i = state_action[0][0] * (state_action[0][2] + gamma * self.value[state_action[0][1]])
                policy_state.append((action,policy_state_i))
            best_action = max(policy_state, key = itemgetter(1))
            # print ("Best Action for state {} is {}").format(state, best_action[0])
            self.policy.append(best_action[0])
    
        #print self.policy
        policy_mat = np.matrix(self.policy)
        policy_mat.shape = (3, 4)
        print ("Policy \n {}").format(policy_mat)
    
    
def main():
    pi = ValueIteration()
    env = MDPGridworldEnv()
    pi.initialize()
    pi.evaluate_policy()
    pi.policy_improvement()


if __name__ == '__main__':
    main()

    # def policy_improvement(self, ):
