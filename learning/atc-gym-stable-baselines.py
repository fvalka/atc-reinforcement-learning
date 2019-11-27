import datetime
import os
import uuid
from multiprocessing import freeze_support

import gym
import yaml
from gym.wrappers import TimeLimit
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
import tensorflow as tf
import numpy as np

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


def learn(multiprocess: bool = True, time_steps: int = int(1e6), record_video: bool = True):
    def callback(locals_, globals_):
        self_ = locals_["self"]

        mean_actions = np.mean(self_.env.get_attr("actions_per_timestep"))
        mean_actions_tf = tf.Summary(value=[tf.Summary.Value(tag='simulation/mean_actions', simple_value=mean_actions)])
        winning_ratio = np.mean(self_.env.get_attr("winning_ratio"))
        winning_ratio_tf = tf.Summary(value=[tf.Summary.Value(tag='simulation/winning_ratio', simple_value=winning_ratio)])
        fps = tf.Summary(value=[tf.Summary.Value(tag='simulation/fps', simple_value=locals_['fps'])])

        locals_['writer'].add_summary(fps, self_.num_timesteps)
        locals_['writer'].add_summary(mean_actions_tf, self_.num_timesteps)
        locals_['writer'].add_summary(winning_ratio_tf, self_.num_timesteps)

        return True

    def video_trigger(step):
        # allow warm-up for video recording
        if not record_video or step < time_steps/3:
            return False

        return step % (int(time_steps/8)) == 0

    log_dir = "../logs/%s/" % datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    log_dir_tensorboard = "../logs/tensorboard/"
    print("Tensorboard log directory: %s" % os.path.abspath(log_dir_tensorboard))

    model_dir = os.path.join(log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

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

    if record_video:
        env = VecVideoRecorder(env, video_dir, video_trigger, video_length=2000)

    hyperparams = {"n_steps": 1024,
                   "nminibatches": 32,
                   "cliprange": 0.3,
                   "gamma": 0.999,
                   "lam": 0.95,
                   "learning_rate": lambda step: LinearSchedule(1.0, initial_p=0.0001, final_p=0.001).value(step),
                   "noptepochs": 4,
                   "ent_coef": 0.01}

    yaml.dump(hyperparams, open(os.path.join(model_dir, "hyperparams.yml"), "w+"))

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir_tensorboard, **hyperparams)
    # model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_steps, callback=callback)

    model.save("%s/PPO2_atc_gym" % model_dir)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    freeze_support()
    learn(time_steps=int(1e6), multiprocess=True, record_video=False)
