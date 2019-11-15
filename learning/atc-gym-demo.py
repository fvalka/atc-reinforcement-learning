import gym
import time
from gym.envs.classic_control.rendering import SimpleImageViewer

import envs.atc.atc_gym

env = gym.make('AtcEnv-v0')
#env.step()

num = 100
t0 = time.time()
for i in range(num):
    env.reset()
    env.render()
t1 = time.time()

print("Finished!")
print("FPS: %f" % (num/(t1-t0)))
