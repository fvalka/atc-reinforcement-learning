import gym
import envs.atc.atc_gym

env = gym.make('AtcEnv-v0')
#env.step()
env.reset()
env.render()