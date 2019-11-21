import os
import time
import uuid
from multiprocessing import freeze_support

import gym
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


def learn(multiprocess=True, normalize=True, time_steps=int(1e6)):
    log_dir = "../logs/%.2f/" % time.time()

    def make_env():
        log_dir_single = "%s/%s/" % (log_dir, uuid.uuid4())
        env = gym.make('AtcEnv-v0')
        os.makedirs(log_dir_single, exist_ok=True)
        env = Monitor(env, log_dir_single, allow_early_resets=True)
        return env

    if multiprocess:
        n_cpu = 16
        env = SubprocVecEnv([lambda: make_env() for i in list(range(n_cpu))])
    else:
        env = DummyVecEnv([lambda: make_env()])

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO2(MlpPolicy, env, verbose=0)
    # model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_steps)

    model_dir = "%s/%s/" % (log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save("%s/PPO2_atc_gym" % model_dir)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(0.1)


if __name__ == '__main__':
    freeze_support()
    learn()
