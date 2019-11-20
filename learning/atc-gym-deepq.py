import gym
import time
from baselines import deepq
from baselines import ppo2

# noinspection PyUnresolvedReferences (is used for registering the atc gym in the OpenAI gym framework)
import envs.atc.atc_gym

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) >= 20
    return is_solved


def main():
    env = gym.make('AtcEnv-v0')
    env.reset()
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model")
    act.save("atc-gym-deepq.pkl")


if __name__ == '__main__':
    main()