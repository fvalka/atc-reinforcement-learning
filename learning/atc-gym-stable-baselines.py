import datetime
import os
import yaml
import uuid
from multiprocessing import freeze_support

import gym
from gym.wrappers import TimeLimit
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


def learn(multiprocess: bool = True, normalize: bool = True, time_steps: int = int(1e6)):
    log_dir = "../logs/%s/" % datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    log_dir_tensorboard = "../logs/tensorboard/"
    print("Tensorboard log directory: %s" % os.path.abspath(log_dir_tensorboard))

    model_dir = "%s/%s/" % (log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        log_dir_single = "%s/%s/" % (log_dir, uuid.uuid4())
        env = gym.make('AtcEnv-v0')
        env = TimeLimit(env, 8000)
        os.makedirs(log_dir_single, exist_ok=True)
        env = Monitor(env, log_dir_single, allow_early_resets=True)
        return env

    n_envs = 8
    if multiprocess:
        env = SubprocVecEnv([lambda: make_env() for i in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: make_env()])

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    hyperparams = {"n_steps": 1024,
                   "nminibatches": 32,
                   "cliprange": 0.4,
                   "gamma": 0.99,
                   "lam": 0.95,
                   "learning_rate": lambda step: LinearSchedule(1.0, initial_p=0.0005, final_p=0.001).value(step),
                   "noptepochs": 20,
                   "ent_coef": 0.01}

    yaml.dump(hyperparams, open(os.path.join(model_dir, "hyperparams.yml"), "w+"))

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir_tensorboard, **hyperparams)
    # model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_steps)

    model.save("%s/PPO2_atc_gym" % model_dir)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    freeze_support()
    learn(time_steps=int(5e6), multiprocess=True)
