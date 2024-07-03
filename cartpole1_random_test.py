import gymnasium as gym
import tensorflow as tf
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

env=gym.make("CartPole-v1",render_mode = 'human')
state_size = len(env.observation_space.low)

test_episodes = 10
reward = []

for i in tqdm.tqdm(range(1, test_episodes+1), ascii = True, unit = 'episode'):
    ep_reward = 0
    s = env.reset()[0]
    #print(s)
    done = False
    
    step = 0
    while done != True:
        s = s.reshape(1,state_size)
        
        action = np.random.choice(env.action_space.n)
        

        s_, r, done, truncated , info = env.step(action)
        done = done or truncated

        s = s_
        ep_reward += r
        step += 1


    print("end of episode", i,"cumulated reward", ep_reward)
    reward.append(ep_reward)

print("max  reward",np.max(reward))
print("min  reward",np.min(reward))
print("mean reward",np.mean(reward))
print("std  reward",np.std(reward))

plt.plot([i for i in range(len(reward))], reward)
plt.xlabel('Episodes')
plt.ylabel('Episode reward')
plt.savefig(r'vizTest.png')