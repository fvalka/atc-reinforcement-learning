from typing import List

import gym
import gym.spaces
import numpy as np
import shapely.geometry as shape
from gym.utils import seeding

from . import model


class AtcGym(gym.Env):
    def __init__(self):
        self._mvas = self._generate_mvas()
        self._runway = self._generate_runway()
        self._airspace = self._generate_airspace(self._mvas, self._runway)

        self._sim_parameters = model.SimParameters(1)

        # action space structure: v, h, phi
        self.action_space = gym.spaces.Box(low=np.array([100, 0, 0]),
                                           high=np.array([300, 40000, 360 - self._sim_parameters.precision]))

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf, phi_rel_runway
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                                high=np.array(
                                                    [50, 50, 40000, 360 - self._sim_parameters.precision, 400, 36000,
                                                     50, 360, 360]))

        self.reward_range = (-100.0, 100.0)

        self.reset()

        self.viewer = None

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
        reward = -0.001

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        reward += self._action_with_reward(self._airplane.action_v, action[0])
        reward += self._action_with_reward(self._airplane.action_h, action[1])
        reward += self._action_with_reward(self._airplane.action_phi, action[2])

        self._airplane.step()

        # check that the plane is above the MVA (minimum vectoring altitude)
        try:
            mva = self._airspace.get_mva_height(self._airplane.x, self._airplane.y)

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

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf, phi_rel_runway
        # FIXME calculate relative angle to FAF phi_rel_faf
        to_faf_x = self._corridor.faf[0] - self._airplane.x
        to_faf_y = self._corridor.faf[1] - self._airplane.y
        d_faf = np.hypot(to_faf_x, to_faf_y)
        phi_rel_faf = np.arctan2(to_faf_y, to_faf_x)
        phi_rel_runway = model.relative_angle(self._runway.phi_to_runway, self._airplane.phi)

        state = np.array([self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi,
                          self._airplane.v, self._airplane.h - mva, d_faf, phi_rel_faf, phi_rel_runway],
                         dtype=np.float32)

        return state, reward, done, {}

    @staticmethod
    def _action_with_reward(func, action):
        try:
            action_taken = func(action)
            if action_taken:
                return -0.01
        except ValueError:
            # invalid action, outside of permissible range
            return -0.1
        return 0.0

    def reset(self):
        """
        Reset the environment

        Creates a new airplane instance
        :return:
        """

        self._airplane = model.Airplane(self._sim_parameters, 0, 10, 16000, 90, 250)

    def render(self, mode='human'):
        """
        Rendering the environments state
        """
        def transform_world_to_screen(coords):
            return [((coord[0] + self._world_x0) * self._scale + self._padding,
                     (coord[1] + self._world_y0) * self._scale + self._padding) for coord in coords]

        if self.viewer is None:
            self._padding = 10
            screen_width = 600

            bbox = self._airspace.get_bounding_box()
            self._world_x0 = bbox[0]
            self._world_y0 = bbox[1]
            world_size_x = bbox[2] - self._world_x0
            world_size_y = bbox[3] - self._world_y0
            self._scale = screen_width / world_size_x
            screen_height = int(world_size_y * self._scale)

            from gym.envs.classic_control import rendering
            from pyglet import gl
            self.viewer = rendering.Viewer(screen_width + 2 * self._padding, screen_height + 2 * self._padding)

            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glEnable(gl.GL_POLYGON_SMOOTH)

            for mva in self._mvas:
                mva_outline = rendering.PolyLine(transform_world_to_screen(mva.area.exterior.coords), True)
                mva_outline.set_linewidth(2)
                mva_outline.set_color(21./256., 79./256., 113./256.)
                self.viewer.add_geom(mva_outline)

        return self.viewer.render(mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _generate_mvas(self) -> List[model.MinimumVectoringAltitude]:
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]
        return mvas

    def _generate_airspace(self, mvas: List[model.MinimumVectoringAltitude], runway: model.Runway) -> model.Airspace:
        airspace = model.Airspace(mvas, runway)
        return airspace

    def _generate_runway(self) -> model.Runway:
        x = 20
        y = 20
        h = 0
        phi = 180
        runway = model.Runway(x, y, h, phi)
        return runway
