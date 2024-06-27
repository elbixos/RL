import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

env=gym.make("CartPole-v1",render_mode = 'human')
state_size = len(env.observation_space.low)

my_model = tf.keras.models.load_model('model_actor.keras')
test_episodes = 10
reward = []
epsilon = 0.05

p_danger = 0.10


for i in tqdm.tqdm(range(1, test_episodes+1), ascii = True, unit = 'episode'):
    ep_reward = 0
    s = env.reset()[0]
    #print(s)
    done = False
    

    while done != True:
        s = s.reshape(1,state_size)
        '''
        if np.random.random() > (1 - epsilon):
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(model.predict(s,verbose=0))
        '''
        action = np.argmax(my_model.predict(s,verbose=0))

        # add a probability of opposite action...
        if np.random.random() < p_danger:
            action = not(action)

        s_, r, done, truncated , info = env.step(action)
        done = done or truncated

        s = s_
        ep_reward += r

    reward.append(ep_reward)

print("max  reward",np.max(reward))
print("min  reward",np.min(reward))
print("mean reward",np.mean(reward))
print("std  reward",np.std(reward))

plt.plot([i for i in range(len(reward))], reward)
plt.xlabel('Episodes')
plt.ylabel('Episode reward')
plt.savefig(r'vizTest.png')