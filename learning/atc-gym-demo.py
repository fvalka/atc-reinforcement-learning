import gym
import time
from gym.envs.classic_control.rendering import SimpleImageViewer

import envs.atc.atc_gym

env = gym.make('AtcEnv-v0')
env.reset()

nextaction = env.action_space.sample()

num = 1000
t0 = time.time()
for i in range(num):
    env.step(nextaction)
    env.render()
    time.sleep(1)
    if i % 20 == 0:
        nextaction = env.action_space.sample()
t1 = time.time()

print("Finished!")
print("FPS: %f" % (num/(t1-t0)))
