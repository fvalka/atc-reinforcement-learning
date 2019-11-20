import os
from multiprocessing import freeze_support

import gym
import time
import matplotlib.pyplot as plt
from stable_baselines.bench import Monitor

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2, ACKTR, results_plotter

import envs.atc.atc_gym


def learn():

    def make_env():
        env = gym.make('AtcEnv-v0')
        return env

    # Create log dir
    log_dir = "../logs/"
    os.makedirs(log_dir, exist_ok=True)
    # multiprocess environment
    n_cpu = 16
    env = SubprocVecEnv([lambda: make_env() for i in list(range(n_cpu))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO2(MlpPolicy, env, verbose=1)
    # model = ACKTR(MlpPolicy, env, verbose=1)
    time_steps = int(1e8)
    model.learn(total_timesteps=time_steps)
    #model.save("PPO2_atc_gym")

    #results_plotter.plot_results(["../logs/"], time_steps, results_plotter.X_TIMESTEPS, "ATC Gym")
    #plt.show()

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(0.1)


if __name__ == '__main__':
    freeze_support()
    learn()
