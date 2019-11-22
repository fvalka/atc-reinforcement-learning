import datetime
import os
import uuid

import gym
import yaml
from gym.wrappers import TimeLimit
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecCheckNan

from learning import hyperparam_optimization


class TuningParameters:
    n_trials: int
    n_timesteps: int
    sampler: str
    pruner: str

    def __init__(self, n_trials: int, n_timesteps: int, sampler: str, pruner: str):
        """
        Parameters for tuning the hypertuner

        :param n_trials: Number of trials to run
        :param n_timesteps: Number of time steps in each trial
        :param sampler: Name of the sampler which is used. Supports: "random", "tpe", "skopt"
        :param pruner: Name of the pruner which is used. Supports: "halving", "median", "none"
        """
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.sampler = sampler
        self.pruner = pruner


def tune(params: TuningParameters):
    """
    Tunes the PPO2 model for ATC-Gym using the provided TuningParameters in optuna

    :param params: Parameters for tuner
    :return: None
    """
    log_dir = "../logs_optimizer/%s/" % datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_dir_tensorboard = "../logs_optimizer/tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    def model_fn(**kwargs):
        """
        Initializes the model with the given hyperparameters.

        The hyperparameters will also be dumped into a YAML file in the log directory.

        :param kwargs: hyperparameters and other arguments for the model
        :return: PPO2 model with the given hyperparameters
        """
        model_id = uuid.uuid4()
        log_dir_model = os.path.join(log_dir, "model", str(model_id))
        os.makedirs(log_dir_model, exist_ok=True)
        yaml.dump(kwargs, open(os.path.join(log_dir_model, "model_parameters.yml"), 'w+'))

        return PPO2(MlpPolicy, make_env(log_dir_env=log_dir_model), verbose=1,
                    tensorboard_log=log_dir_tensorboard, **kwargs)

    def make_env(n_envs=1, normalize=True, multiprocess=False, log_dir_env=None):
        """
        Initializes an OpenAI Gym environment for training and evaluation.

        :param n_envs: Number of parallel environments to initialize
        :param normalize: Normalization of state values
        :param multiprocess: Use multi processing with the SubprocVecEnv instead of DummyVecEnv. Not recommended.
        :param log_dir_env: Parent directory of the environments log directory
        :return:
        """
        def init_env(log_dir_env):
            if log_dir_env is None:
                log_dir_env = os.path.join(log_dir, "env_direct")
                os.makedirs(log_dir_env, exist_ok=True)
            log_dir_single = os.path.join(log_dir_env, str(uuid.uuid4()))
            env = gym.make('AtcEnv-v0')
            env = TimeLimit(env, 8000)
            os.makedirs(log_dir_single, exist_ok=True)
            env = Monitor(env, log_dir_single, allow_early_resets=True)
            return env

        if multiprocess:
            env = SubprocVecEnv([lambda: init_env(log_dir_env) for i in range(n_envs)])
        else:
            env = DummyVecEnv([lambda: init_env(log_dir_env) for i in range(n_envs)])

        env = VecCheckNan(env, raise_exception=True)

        if normalize:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

        return env

    param_log_file = os.path.join(log_dir, "tuning_parameters.yml")
    yaml.dump(params, open(param_log_file, 'w+'))

    data_frame = hyperparam_optimization.hyperparam_optimization("ppo2", model_fn, make_env,
                                                                 n_trials=params.n_trials,
                                                                 n_timesteps=params.n_timesteps,
                                                                 sampler_method=params.sampler,
                                                                 pruner_method=params.pruner,
                                                                 n_jobs=16)

    report_name = "report_{}-trials-{}-{}-{}.csv".format(params.n_trials, params.n_timesteps,
                                                         params.sampler, params.pruner)

    log_path = os.path.join(log_dir, report_name)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    data_frame.to_csv(log_path)


if __name__ == '__main__':
    # freeze_support()
    parameters = TuningParameters(n_trials=20, n_timesteps=int(1e5), sampler="tpe", pruner="halving")
    tune(parameters)
