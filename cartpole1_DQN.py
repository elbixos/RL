# -*- coding: utf-8 -*-
"""cartpole_DQN1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1M8BNVL4_MuvQ0O5ZSGuBE9i89gf1tJX3
"""

#!pip install gymnasium

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import math

class Model:
    def __init__(self, input_dim, output_dim, lr):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        # Using Target variant
        self.Qpolicy = self.create()
        self.Qtarget = self.create()
        self.Qtarget.set_weights(self.Qpolicy.get_weights())

    def create(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(self.output_dim, activation = 'linear'))
        model.compile(optimizer = Adam(learning_rate = self.lr), loss = "mse", metrics = ['accuracy'])
        return model

class DQNAgent:
    def __init__(self, lr = 0.001, gamma = 0.99, epsilon = 1, decay_coe = 0.99975, min_eps = 0.05,
                 batch_size = 128, memory_size = 10000, episodes = 5000, C = 5,render = None):

        if render == None:
            self.env = gym.make('CartPole-v1')
        else :
            self.env = gym.make('CartPole-v1',render_mode = 'human')
        #obs, info = self.env.reset(seed=42)

        self.states = len(self.env.observation_space.low)
        self.n_actions = self.env.action_space.n

        self.actions = [i for i in range(self.n_actions)]

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_coe = decay_coe
        self.min_eps = min_eps
        self.episodes = episodes
        self.batch_size = batch_size
        self.D = deque(maxlen = memory_size) # replay memory
        self.C = C

        self.terminal_state = False # end of the episode
        self.target_counter = 0

        # Plot data
        self.timestep = self.episodes / 10
        self.history = []
        self.reward_data = []
        self.epsilon_data = []

        self.model = Model(self.states, self.n_actions, self.lr)
        # Smooth epsilon
        self.a = 0.5
        self.b = 0.25
        self.c = 0.01

    def state_shape(self,states):
        states = np.array(states)

        return states.reshape(-1)

    def decrement_epsilon(self, time):
        '''
        if self.epsilon > self.min_eps:
            self.epsilon *= self.decay_coe
        else:
            self.epsilon = self.min_eps
        
        '''
        s_time = (time - self.a*self.episodes) / (self.b*self.episodes)
        cosh = np.cosh(math.exp(-s_time))
        #self.epsilon = 1 - (1/cosh + (time*self.c/self.episodes))
        self.epsilon = (1-1/cosh)*(1-self.c) + self.c


    def update_D(self, s, a, r, s_, done):
        #print ("YOUHOU", self.state_shape(s))
        new_data = [self.state_shape(s), a, r, self.state_shape(s_), done]
        #print("youhou",new_data)
        self.D.append(new_data)

    def choose_action(self, states):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.choice(self.actions)
        else:
            states = self.state_shape(states)
            states = states.reshape(1,self.states)
            action = np.argmax(self.model.Qpolicy.predict(states,verbose=0))

        return action

    def minibatch(self):
        return random.sample(self.D, self.batch_size)

    def graphs(self, episode):
        f1 = plt.figure(1)
        plt.plot([i for i in range(len(self.reward_data))], self.reward_data)
        plt.ylabel('Score per episode')
        plt.xlabel('Episodes')
        plt.savefig(r'viz_reward - e{}v2.png'.format(episode), dpi = 500)

        f2 = plt.figure(2)
        plt.plot([i for i in range(len(self.epsilon_data))], self.epsilon_data)
        plt.ylabel('Epsilon')
        plt.xlabel('Episodes')
        plt.savefig(r'viz_epsilon - e{}v2.png'.format(episode), dpi = 500)

        f3 = plt.figure(3)
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.savefig(r'viz_loss - e{}v2.png'.format(episode), dpi = 500)

    def train(self):
        # X - states passed to the NN, y - target

        X, y = [], []

        if len(self.D) >= self.batch_size:
            SARS = self.minibatch()


            s = np.array([row[0] for row in SARS])

            qvalue = self.model.Qpolicy.predict(s,verbose=0)

            s_ = np.array([row[3] for row in SARS])
            future_qvalue = self.model.Qtarget.predict(s_,verbose=0)
            #print("future_qvalue",future_qvalue)


            for index, (state, action, reward, state_, done) in enumerate(SARS):
                #print()
                #print("enum",index)
                if done == True:
                    Qtarget = reward
                else:
                    Qtarget = reward + self.gamma * np.max(future_qvalue[index])

                # To update the prediction of the model for the chosen action,
                # let's say that the target values are all correct for all
                # actions except the one we used.


                #print("predicted",qvalue[index])
                qcurr = qvalue[index]
                #print("predicted also",qcurr)
                qcurr[action] = Qtarget
                #print("target with action",action," modified",qcurr)
                X.append(state)
                y.append(qcurr)

            #X, y = np.array(X).reshape(1,self.batch_size,1,self.states), np.array(y).reshape(1,self.batch_size, 1, self.n_actions)
            X, y = np.array(X).reshape(self.batch_size,self.states), np.array(y).reshape(self.batch_size, self.n_actions)

            loss = self.model.Qpolicy.fit(X, y, batch_size = self.batch_size, shuffle = False, verbose = 0)
            self.history.append(loss.history['loss'][0])


            if self.terminal_state:
                self.target_counter+=1

            # C -> target network update frequency
            if self.target_counter > self.C:
                self.model.Qtarget.set_weights(self.model.Qpolicy.get_weights())
                self.target_counter = 0


    def training(self):
        timestep_reward = 0

        for episode in tqdm(range(1, self.episodes+1), ascii = True, unit = 'episode'):
            s = self.env.reset()
            s = s[0]

            done = False
            score = 0
            while done != True:
                a = self.choose_action(s)

                s_, r, done, truncated, info= self.env.step(a)
                
                # consider done True if failure or steps > 500
                done = done or truncated

                #print("state",s,"action", a,("state'"), s_,"reward", r,"done", done)

                # Update
                self.terminal_state = done
                self.update_D(s,a,r,s_,done)
                #print(self.D)

                self.train()

                s = s_
                score += r

            self.decrement_epsilon(episode)

            # UPDATE
            self.reward_data.append(score)
            self.epsilon_data.append(self.epsilon)

            if episode % self.timestep == 0:
                self.graphs(episode)
                self.model.Qpolicy.save(r'qpol_model{}.keras'.format(episode))



        self.graphs(episode)
        self.model.Qpolicy.save('qpol_model.keras')

    def test(self, model_name,test_episodes = 100):
        model = tf.keras.models.load_model('{}'.format(model_name))
        reward = []
        epsilon = 0.05
        for i in tqdm(range(1, test_episodes+1), ascii = True, unit = 'episode'):
            ep_reward = 0
            s = self.env.reset()
            s= s[0]
            #print(s)
            done = False

            while done != True:
                s = s.reshape(1,self.states)
                '''
                if np.random.random() > (1 - epsilon):
                    action = np.random.choice(self.actions)
                else:
                    action = np.argmax(model.predict(s,verbose=0))
                '''
                action = np.argmax(model.predict(s,verbose=0))

                s_, r, done, truncated , info = self.env.step(action)
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

dqn = DQNAgent(episodes = 1000,render=True)
#dqn.training()

dqn.test("qpol_model900.keras",test_episodes = 10)