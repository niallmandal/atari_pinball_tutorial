import gym

env = gym.make('VideoPinball-v0')

print(env.action_space)

frame = env.reset()
env.render()

is_done = False
while not is_done:
    frame, rew, is_done, _ = env.step(env.action_space.sample())
    env.render()
