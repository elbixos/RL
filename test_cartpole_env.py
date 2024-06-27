import gymnasium as gym

env = gym.make('CartPole-v1',render_mode = 'human')

for episode in range(10):
    s = env.reset()
    s = s[0]
    print(s)


