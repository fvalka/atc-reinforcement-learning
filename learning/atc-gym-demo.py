import gym
import time

# noinspection PyUnresolvedReferences (is used for registering the atc gym in the OpenAI gym framework)
import envs.atc.atc_gym

env = gym.make('AtcEnv-v0')
env.reset()

nextaction = env.action_space.sample()

num = 1000
t0 = time.time()
for i in range(num):
    state, reward, done, info = env.step(nextaction)
    env.render()
    #time.sleep(1)
    if i % 20 == 0:
        nextaction = env.action_space.sample()
    print("reward: %s || done: %s" % (state, done))
t1 = time.time()

print("Finished!")
print("FPS: %f" % (num/(t1-t0)))
