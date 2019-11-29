import gym
import time

# noinspection PyUnresolvedReferences (is used for registering the atc gym in the OpenAI gym framework)
import envs.atc.atc_gym

env = gym.make('AtcEnv-v0')
env.reset()

nextaction = env.action_space.sample()

num = 100000
t0 = time.time()
for i in range(num):
    state, reward, done, info = env.step(nextaction)
t1 = time.time()

print("Finished!")
print("FPS: %f" % (num/(t1-t0)))
