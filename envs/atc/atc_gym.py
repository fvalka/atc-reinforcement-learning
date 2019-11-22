from typing import List

import gym
import gym.spaces
import numpy as np
import shapely.geometry as shape
from gym.envs.classic_control import rendering
from gym.utils import seeding

from envs.atc.rendering import Label
from envs.atc.themes import ColorScheme
from . import model


class AtcGym(gym.Env):
    def __init__(self, sim_parameters=model.SimParameters(1)):
        self._mvas = self._generate_mvas()
        self._runway = self._generate_runway()
        self._airspace = self._generate_airspace(self._mvas, self._runway)
        self._faf_mva = self._airspace.get_mva_height(self._runway.corridor.faf[0][0], self._runway.corridor.faf[1][0])

        self._sim_parameters = sim_parameters

        self.normalization_offset = np.array([100, 0, 0])
        self.normalization_factor = np.array([200, 40000, 360 - self._sim_parameters.precision])

        # action space structure: v, h, phi
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0]),
                                           high=np.array([1, 1, 1]))

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf, phi_rel_runway
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                                high=np.array(
                                                    [110, 50, 40000, 360 - self._sim_parameters.precision, 400, 36000,
                                                     50, 360, 360]))

        self.reward_range = (-300.0, 100.0)

        self.done = True
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
        self.done = False
        reward = -0.005 * self._sim_parameters.timestep

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        def denormalized_action(index):
            return action[index] * self.normalization_factor[index] + self.normalization_offset[index]

        reward += self._action_with_reward(self._airplane.action_v, denormalized_action(0))
        reward += self._action_with_reward(self._airplane.action_h, denormalized_action(1))
        reward += self._action_with_reward(self._airplane.action_phi, denormalized_action(2))

        self._airplane.step()

        # check that the plane is above the MVA (minimum vectoring altitude)
        try:
            mva = self._airspace.get_mva_height(self._airplane.x, self._airplane.y)

            if self._airplane.h < mva:
                # Airplane has descended below the minimum vectoring altitude
                reward = -200
                self.done = True
            # else:
            # reward += (self._airplane.h - mva)/40000 * 0.0004
        except ValueError:
            # Airplane has left the airspace
            self.done = True
            reward = -50
            mva = 0  # dummy value outside of range so that the MVA is set for the last state

        if self._runway.inside_corridor(self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi):
            # GAME WON!
            reward = 100
            self.done = True

        state = self._get_state()
        self.state = state

        # advanced award for approach sector location
        reward_faf = 1 / np.maximum(np.power(self._d_faf, 0.2), 1)
        reward_app_angle = np.power(
            1 - np.abs(model.relative_angle(self._runway.phi_to_runway, self._phi_rel_faf)) / 180, 1.5)
        reward += reward_faf * reward_app_angle * 2.0

        return state, reward, self.done, {}

    def _get_state(self):
        try:
            mva = self._airspace.get_mva_height(self._airplane.x, self._airplane.y)
        except ValueError:
            # Airplane left airspace, simulation must be done, otherwise this is a bug
            if not self.done:
                raise AssertionError("Mva not found but simulation is not done yet")
            mva = 0
        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf, phi_rel_runway
        to_faf_x = self._runway.corridor.faf[0][0] - self._airplane.x
        to_faf_y = self._runway.corridor.faf[1][0] - self._airplane.y
        phi_rel_runway = model.relative_angle(self._runway.phi_to_runway, self._airplane.phi)
        self._d_faf = np.sqrt(np.hypot(to_faf_x, to_faf_y) ** 2 + self._faf_mva ** 2)
        self._phi_rel_faf = np.degrees(np.arctan2(to_faf_y, to_faf_x))
        state = np.array([self._airplane.x, self._airplane.y, self._airplane.h, self._airplane.phi,
                          self._airplane.v, self._airplane.h - mva, self._d_faf, self._phi_rel_faf, phi_rel_runway],
                         dtype=np.float32)
        return state

    @staticmethod
    def _action_with_reward(func, action):
        try:
            action_taken = func(action)
            if action_taken:
                return -0.005
                # return 0.0
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
        self.done = False
        self._airplane = model.Airplane(self._sim_parameters, "FLT01", 9, 30, 16000, 90, 250)
        self.state = self._get_state()
        return self.state

    def render(self, mode='human'):
        """
        Rendering the environments state
        """
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

            background = rendering.FilledPolygon([(0, 0), (0, screen_height + 2 * self._padding),
                                                  (screen_width + 2 * self._padding, screen_height + 2 * self._padding),
                                                  (screen_width + 2 * self._padding, 0)])
            background.set_color(*ColorScheme.background_inactive)
            self.viewer.add_geom(background)

            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glEnable(gl.GL_POLYGON_SMOOTH)

            self._render_mvas()
            self._render_runway()
            self._render_faf()
            self._render_approach()

        self._render_airplane(self._airplane)

        return self.viewer.render(mode == 'rgb_array')

    def _render_airplane(self, airplane: model.Airplane):
        render_size = 4
        vector = self._screen_vector(airplane.x, airplane.y)
        corner_vector = np.array([[0], [render_size]])
        corner_top_right = np.dot(model.rot_matrix(45), corner_vector) + vector
        corner_bottom_right = np.dot(model.rot_matrix(135), corner_vector) + vector
        corner_bottom_left = np.dot(model.rot_matrix(225), corner_vector) + vector
        corner_top_left = np.dot(model.rot_matrix(315), corner_vector) + vector

        symbol = rendering.PolyLine([corner_top_right, corner_bottom_right, corner_bottom_left, corner_top_left], True)
        symbol.set_color(*ColorScheme.airplane)
        symbol.set_linewidth(2)
        self.viewer.add_onetime(symbol)

        label_pos = np.dot(model.rot_matrix(135), 2 * corner_vector) + vector
        render_altitude = round(airplane.h / 100)
        render_speed = round(airplane.v / 10)
        render_text = "%d  %d" % (render_altitude, render_speed)
        label_name = Label(airplane.name, x=label_pos[0][0], y=label_pos[1][0])
        label_details = Label(render_text, x=label_pos[0][0], y=label_pos[1][0] - 15)
        self.viewer.add_onetime(label_name)
        self.viewer.add_onetime(label_details)

        n = len(airplane.position_history)
        for i in range(n - 5, max(0, n - 25), -1):
            if i % 5 == 0:
                circle = rendering.make_circle(radius=2, res=12)
                screen_vector = self._screen_vector(airplane.position_history[i][0], airplane.position_history[i][1])
                transform = rendering.Transform(translation=(screen_vector[0][0], screen_vector[1][0]))
                circle.add_attr(transform)
                circle.set_color(*ColorScheme.airplane)
                self.viewer.add_onetime(circle)

    def _render_approach(self):
        iaf_x = self._runway.corridor.iaf[0][0]
        iaf_y = self._runway.corridor.iaf[1][0]
        dashes = 48
        runway_vector = self._screen_vector(self._runway.x, self._runway.y)
        runway_iaf = self._screen_vector(iaf_x - self._runway.x, iaf_y - self._runway.y)
        for i in range(int(dashes / 2 + 1)):
            start = runway_vector + runway_iaf / dashes * 2 * i
            end = runway_vector + runway_iaf / dashes * (2 * i + 1)
            dash = rendering.PolyLine([start, end], False)
            dash.set_color(*ColorScheme.lines_info)
            self.viewer.add_geom(dash)

    def _render_faf(self):
        faf_screen_render_size = 6

        faf_x = self._runway.corridor.faf[0][0]
        faf_y = self._runway.corridor.faf[1][0]
        faf_vector = self._screen_vector(faf_x, faf_y)

        corner_vector = np.array([[0], [faf_screen_render_size]])
        corner_top = faf_vector + corner_vector
        corner_right = np.dot(model.rot_matrix(121), corner_vector) + faf_vector
        corner_left = np.dot(model.rot_matrix(242), corner_vector) + faf_vector

        poly_line = rendering.PolyLine([corner_top, corner_right, corner_left], True)
        poly_line.set_color(*ColorScheme.lines_info)
        poly_line.set_linewidth(2)
        self.viewer.add_geom(poly_line)

    def _render_mvas(self):
        def transform_world_to_screen(coords):
            return [((coord[0] + self._world_x0) * self._scale + self._padding,
                     (coord[1] + self._world_y0) * self._scale + self._padding) for coord in coords]

        for mva in self._mvas:
            coordinates = transform_world_to_screen(mva.area.exterior.coords)

            fill = rendering.FilledPolygon(coordinates)
            fill.set_color(*ColorScheme.background_active)
            self.viewer.add_geom(fill)

            outline = rendering.PolyLine(coordinates, True)
            outline.set_linewidth(1)
            outline.set_color(*ColorScheme.mva)
            self.viewer.add_geom(outline)

    def _render_runway(self):
        runway_length = 1.7 * self._scale
        runway_to_threshold_vector = \
            np.dot(model.rot_matrix(self._runway.phi_from_runway), np.array([[0], [runway_length / 2]]))
        runway_vector = self._screen_vector(self._runway.x, self._runway.y)
        runway_line = rendering.PolyLine(
            [runway_vector - runway_to_threshold_vector, runway_vector + runway_to_threshold_vector], False)
        runway_line.set_linewidth(5)
        runway_line.set_color(*ColorScheme.runway)
        self.viewer.add_geom(runway_line)

    def _screen_vector(self, x, y):
        return np.array([
            [(x + self._world_x0) * self._scale],
            [(y + self._world_y0) * self._scale]
        ])

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
        phi = 130
        runway = model.Runway(x, y, h, phi)
        return runway
