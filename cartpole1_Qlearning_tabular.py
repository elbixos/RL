import gymnasium as gym
import numpy as np
import time # to get the time
import math # needed for calculations
import matplotlib
import matplotlib.pyplot as plt

# Compute index of discretised state in the qtable
def returnIndexState(state, lowerBounds,upperBounds,numberOfBins):
        position =      state[0]
        velocity =      state[1]
        angle    =      state[2]
        angularVelocity=state[3]
         
        cartPositionBin=np.linspace(lowerBounds[0],upperBounds[0],numberOfBins[0])
        cartVelocityBin=np.linspace(lowerBounds[1],upperBounds[1],numberOfBins[1])
        poleAngleBin=np.linspace(lowerBounds[2],upperBounds[2],numberOfBins[2])
        poleAngleVelocityBin=np.linspace(lowerBounds[3],upperBounds[3],numberOfBins[3])
         
        indexPosition=np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        indexAngle=np.maximum(np.digitize(state[2],poleAngleBin)-1,0)
        indexAngularVelocity=np.maximum(np.digitize(state[3],poleAngleVelocityBin)-1,0)
         
        return tuple([indexPosition,indexVelocity,indexAngle,indexAngularVelocity])

def selectAction(state,epsilon, Qmatrix,lowerBounds,upperBounds,numberOfBins ,actionNumber):
        
        randomNumber=np.random.random()
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(actionNumber)            
         
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qmatrix[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            return np.random.choice(np.where(Qmatrix[returnIndexState(state, lowerBounds,upperBounds,numberOfBins)]==np.max(Qmatrix[returnIndexState(state, lowerBounds,upperBounds,numberOfBins)]))[0])

# 1. Load Environment and Q-table structure
env = gym.make('CartPole-v1',render_mode='human')

# Observations :
#   Cart_pos (-4.8, 4.8) but terminate outside (-2.4,2.4)
#   Cart_vel (-Inf, Inf)
#   Pole_angle (-0.418rad (-24°), 0.418rad (24°)) but terminate outside (-0.2095,0.2095)
#   Pole_vel (-Inf, Inf)

# Useful for discretisation
upperBounds=env.observation_space.high
lowerBounds=env.observation_space.low
cartVelocityMin=-3
cartVelocityMax=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
upperBounds[1]=cartVelocityMax
upperBounds[3]=poleAngleVelocityMax
lowerBounds[1]=cartVelocityMin
lowerBounds[3]=poleAngleVelocityMin

numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]


#print(env.observation_space)
print(env.action_space.n)
Qmatrix=np.random.uniform(low=0, high=1, size=(numberOfBins[0],numberOfBins[1],numberOfBins[2],numberOfBins[3],env.action_space.n))

# 2. Parameters of Q-learning
alpha=0.15
gamma=0.95
epsilon=0.8
numberEpisodes=10000

sumRewardsEpisode = []


for indexEpisode in range(numberEpisodes):    
    # list that stores rewards per episode - this is necessary for keeping track of convergence 
    rewardsEpisode=[]
    # reset the environment at the beginning of every episode
    (stateS,_)=env.reset()
    stateS=list(stateS)

    print("Simulating episode {}".format(indexEpisode))
        
        
    # here we step from one state to another
    # this will loop until a terminal state is reached
    terminalState=False
    while not terminalState:
        # return a discretized index of the state
        epsilon *=0.9999    
        stateSIndex=returnIndexState(stateS, lowerBounds,upperBounds,numberOfBins)
            
        # select an action on the basis of the current state, denoted by stateS
        actionA = selectAction(stateS,epsilon, Qmatrix, lowerBounds,upperBounds,numberOfBins,env.action_space.n)
            
            
        # here we step and return the state, reward, and boolean denoting if the state is a terminal state
        # prime means that it is the next state
        (stateSprime, reward, terminalState,_,_) = env.step(actionA)          
            
        rewardsEpisode.append(reward)
            
        stateSprime=list(stateSprime)
            
        stateSprimeIndex=returnIndexState(stateSprime, lowerBounds,upperBounds,numberOfBins)
            
        # return the max value, we do not need actionAprime...
        QmaxPrime=np.max(Qmatrix[stateSprimeIndex])                                               
                                        
        if not terminalState:
            # stateS+(actionA,) - we use this notation to append the tuples
            # for example, for stateS=(0,0,0,1) and actionA=(1,0)
            # we have stateS+(actionA,)=(0,0,0,1,0)
            error=reward+gamma*QmaxPrime-Qmatrix[stateSIndex+(actionA,)]
            Qmatrix[stateSIndex+(actionA,)]=Qmatrix[stateSIndex+(actionA,)]+alpha*error
        else:
            # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0 
            error=reward-Qmatrix[stateSIndex+(actionA,)]
            Qmatrix[stateSIndex+(actionA,)]=Qmatrix[stateSIndex+(actionA,)]+alpha*error
            
        # set the current state to the next state                    
        stateS=stateSprime

    print("epsilon",epsilon,"Sum of rewards {}".format(np.sum(rewardsEpisode)))        
    sumRewardsEpisode.append(np.sum(rewardsEpisode))

env.close() 

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(sumRewardsEpisode,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.yscale('log')
plt.show()
plt.savefig('convergence.png')