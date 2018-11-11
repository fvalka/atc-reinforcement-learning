import numpy as np
import math
import gym
import gym.spaces
import simulation.model as model
from gym.utils import seeding
import shapely.geometry as shape


class AtcGym(gym.Env):
    def __init__(self) -> None:
        self._mvas = self._generate_mvas()
        self._airspace = self._generate_airspace(self._mvas)
        self._runway = self._generate_runway(self._airspace)
        self._corridor = self._generate_corridor(self._runway)

        self._sim_parameters = model.SimParameters(1)

        # action space structure: v, h, phi
        self.action_space = gym.spaces.Box(low=np.array([100, 0, 0]),
                                           high=np.array([300, 40000, 360 - self._sim_parameters.precision]))

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                                                high=np.array([50, 50, 36000, 360, 400, 36000, 50, 360]))

        self.reward = 1.0
        self.reward_range = (-100.0, 100.0)

    def seed(self, seed=None):
        """
        Seeds the environments pseudo random number generator

        :param seed: A predefined seed for the PRNG
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """

        :param action: Action in format: v, h, phi
        :return:
        """
        done = False
        reward = 0

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        reward += self._action_with_reward(self._airplane.action_v, action[0])
        reward += self._action_with_reward(self._airplane.action_h, action[1])
        reward += self._action_with_reward(self._airplane.action_phi, action[2])

        self._airplane.step()

        # check that the plane is above the MVA (minimum vectoring altitude)
        try:
            mva = self._airspace.get_mva(self._airplane.x, self._airplane.y)

            if self._airplane.h < mva:
                done = True
                reward = -100
        except ValueError:
            # Airplane has left the airspace
            done = True
            reward = -100
            mva = 0  # dummy value outside of range so that the MVA is set for the last state

        if self._corridor.inside_corridor(self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi):
            # GAME WON!
            reward = 100
            done = True

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf
        # FIXME calculate relative angle to FAF phi_rel_faf
        state = np.array([self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi,
                          self._airplane.v, self._airplane.h - mva, 0.0, 0.0], dtype=np.float32)

        return state, reward, done, {}

    @staticmethod
    def _action_with_reward(func, action):
        try:
            action_taken = func(action)
            if action_taken:
                return -0.001
        except ValueError:
            return -0.01

    def reset(self):
        """
        Reset the environment
        :return:
        """

        self._airplane = model.Airplane(self._sim_parameters, 0, 10, 16000, 90, 250)

    def render(self, mode='human'):
        """
        Rendering the environments state
        """
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _generate_mvas(self):
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]
        return mvas

    def _generate_airspace(self, mvas):
        airspace = model.Airspace(mvas)
        return airspace

    def _generate_runway(self, airspace):
        x = 20
        y = 20
        h = 0
        phi = 180
        runway = model.Runway(x, y, h, phi, airspace)
        return runway

    def _generate_corridor(self, runway):
        corridor = model.Corridor(runway)
        return corridor


env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
