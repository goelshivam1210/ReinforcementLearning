import time
import sys
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv

env = gym.make('MDPGridworld-v0')

env = MDPGridworldEnv()

print env.observation_space
print env.action_space

print env.observation_space.n
print env.action_space.n

time,sleep(2)

for i_episode in range(20):
	obs = env.reset()
	for t in range(100):
		env.reader()

		action = env.action_space.sample()
		obs2, reward, terminal, _ = env.step(action)
		if terminal:
			env.reader()
			print("Episode finished after {} timesteps".format(t+1))
			break
			