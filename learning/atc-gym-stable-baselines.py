import datetime
import os
import time
import uuid
from multiprocessing import freeze_support

import gym
from gym.wrappers import TimeLimit
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


def learn(multiprocess: bool = True, normalize: bool = True, time_steps: int = int(1e6)):
    log_dir = "../logs/%s/" % datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def make_env():
        log_dir_single = "%s/%s/" % (log_dir, uuid.uuid4())
        env = gym.make('AtcEnv-v0')
        env = TimeLimit(env, 8000)
        os.makedirs(log_dir_single, exist_ok=True)
        env = Monitor(env, log_dir_single, allow_early_resets=True)
        return env

    n_envs = 16
    if multiprocess:
        env = SubprocVecEnv([lambda: make_env() for i in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: make_env() for i in range(n_envs)])

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    log_dir_tensorboard = "../logs/tensorboard/"
    print("Tensorboard log directory: %s" % os.path.abspath(log_dir_tensorboard))
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir_tensorboard,
                 n_steps=2048, nminibatches=32, gamma=0.99, lam=0.98, noptepochs=4, ent_coef=0.001)
    # model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_steps)

    model_dir = "%s/%s/" % (log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.save("%s/PPO2_atc_gym" % model_dir)

    obs = env.reset()
    for i in range(3000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    freeze_support()
    learn(time_steps=int(1e8), multiprocess=True)
