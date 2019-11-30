import datetime
import os
import uuid
from multiprocessing import freeze_support

import gym
import yaml
from gym.wrappers import TimeLimit
from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
import stable_baselines.sac.policies as sacpolicies
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
import tensorflow as tf
import numpy as np

# noinspection PyUnresolvedReferences
import envs.atc.atc_gym


class ModelFactory:
    hyperparams: dict

    def build(self, env, log_dir):
        pass


def learn(model_factory: ModelFactory, multiprocess: bool = True, time_steps: int = int(1e6),
          record_video: bool = True):
    def callback(locals_, globals_):
        self_ = locals_["self"]

        mean_actions = np.mean(self_.env.get_attr("actions_per_timestep"))
        mean_actions_tf = tf.Summary(value=[tf.Summary.Value(tag='simulation/mean_actions', simple_value=mean_actions)])
        winning_ratio = np.mean(self_.env.get_attr("winning_ratio"))
        winning_ratio_tf = tf.Summary(
            value=[tf.Summary.Value(tag='simulation/winning_ratio', simple_value=winning_ratio)])
        locals_['writer'].add_summary(mean_actions_tf, self_.num_timesteps)
        locals_['writer'].add_summary(winning_ratio_tf, self_.num_timesteps)

        if isinstance(model_factory, PPO2ModelFactory):
            fps = tf.Summary(value=[tf.Summary.Value(tag='simulation/fps', simple_value=locals_['fps'])])
            mean_length = np.mean([info["l"] for info in locals_["ep_infos"]])
            mean_length_tf = tf.Summary(
                value=[tf.Summary.Value(tag='simulation/mean_episode_length', simple_value=mean_length)])
            locals_['writer'].add_summary(fps, self_.num_timesteps)
            locals_['writer'].add_summary(mean_length_tf, self_.num_timesteps)
        return True

    def video_trigger(step):
        # allow warm-up for video recording
        if not record_video or step < time_steps / 3:
            return False

        return step % (int(time_steps / 8)) == 0

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
        env = TimeLimit(env, 4000)
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

    model = model_factory.build(env, log_dir_tensorboard)

    yaml.dump(model_factory.hyperparams, open(os.path.join(model_dir, "hyperparams.yml"), "w+"))

    # model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=time_steps, callback=callback)

    model.save("%s/PPO2_atc_gym" % model_dir)

    # render trained model actions on screen and to file
    eval_observations_file = open(os.path.join(model_dir, "evaluation.csv"), 'a+')
    new_env = gym.make('AtcEnv-v0')
    obs = new_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = new_env.step(action)
        original_state = info["original_state"]
        eval_observations_file.write("%.2f, %.2f, %.0f, %.1f\n" %
                                     (original_state[0], original_state[1], original_state[2], original_state[3]))
        new_env.render()
        if done:
            obs = new_env.reset()


class PPO2ModelFactory(ModelFactory):

    def __init__(self):
        self.hyperparams = {"n_steps": 1024,
                            "nminibatches": 32,
                            "cliprange": 0.4,
                            "gamma": 0.993,
                            "lam": 0.95,
                            "learning_rate": LinearSchedule(1.0, initial_p=0.0002, final_p=0.001).value,
                            "noptepochs": 4,
                            "ent_coef": 0.007}

    def build(self, env, log_dir):
        return PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir, **self.hyperparams)


class SACModelFactory(ModelFactory):

    def __init__(self):
        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size": 1000000,
            "batch_size": 256,
            "ent_coef": "auto",
            "gamma": 0.99,
            "train_freq": 1,
            "tau": 0.005,
            "gradient_steps": 1,
            "learning_starts": 1000
        }

    def build(self, env, log_dir):
        return SAC(sacpolicies.MlpPolicy, env, verbose=1, tensorboard_log=log_dir, **self.hyperparams)


if __name__ == '__main__':
    freeze_support()
    learn(PPO2ModelFactory(), time_steps=int(24e6), multiprocess=True, record_video=False)
