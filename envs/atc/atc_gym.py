import gym
import gym.spaces
import numpy as np
from gym.envs.classic_control import rendering
from gym.utils import seeding

from envs.atc.rendering import Label
from envs.atc.themes import ColorScheme
from . import model
from. import scenarios


class AtcGym(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, sim_parameters=model.SimParameters(1), scenario=scenarios.LOWW()):
        self.last_reward = 0
        self.total_reward = 0

        self._sim_parameters = sim_parameters

        self._mvas = scenario.mvas
        self._runway = scenario.runway
        self._airspace = scenario.airspace
        self._faf_mva = self._airspace.get_mva_height(self._runway.corridor.faf[0][0], self._runway.corridor.faf[1][0])

        bbox = self._airspace.get_bounding_box()
        self._world_x_min = bbox[0]
        self._world_y_min = bbox[1]
        self._world_x_max = bbox[2]
        self._world_y_max = bbox[3]
        world_x_length = self._world_x_max - self._world_x_min
        world_y_length = self._world_y_max - self._world_y_min
        world_max_distance = np.hypot(world_x_length, world_y_length)

        self.done = True
        self.reset()
        self.viewer = None

        self.normalization_action_offset = np.array([self._airplane.v_min, 0, 0])
        self.normalization_action_factor = np.array([self._airplane.v_max - self._airplane.v_min,
                                                     self._airplane.h_max,
                                                     360])

        # action space structure: v, h, phi
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]),
                                           high=np.array([1, 1, 1]))

        self.normalization_state_min = np.array([
            self._world_x_min,
            self._world_y_min,
            0,
            0,
            self._airplane.v_min,
            0,
            0,
            -180,
            -180])
        self.normalization_state_max = np.array([
            world_x_length,  # x position in nautical miles
            world_y_length,  # y positon in nautical miles
            self._airplane.h_max,  # maximum altitude, h in feet
            360,  # airplane heading, phi in degrees
            self._airplane.v_max - self._airplane.v_min,  # airplane speed, v in knots
            self._airplane.h_max,  # height above the current mva in feet
            world_max_distance,  # distance to FAF in nautical miles
            360,  # phi relative to the FAF in degrees (-180, 180)
            360  # phi relative to the runway in degrees (-180, 180)
        ])

        # observation space: x, y, h, phi, v, h-mva, d_faf, phi_rel_faf, phi_rel_runway
        self.observation_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]),
                                                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

        self.reward_range = (-300.0, 1000.0)

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
        Perform a single step in the simulation

        :param action: Action in format: v, h, phi
        :return: Reward obtained from this step
        """
        self.done = False
        reward = -0.05 * self._sim_parameters.timestep

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        def denormalized_action(index):
            return action[index] * self.normalization_action_factor[index] / 2 + \
                   self.normalization_action_factor[index] / 2 + \
                   self.normalization_action_offset[index]

        reward += self._action_with_reward(self._airplane.action_v, denormalized_action(0))
        #reward += self._action_with_reward(self._airplane.action_h, denormalized_action(1))
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
            reward = 8000
            self.done = True

        state = self._get_state()
        self.state = state

        # Reward shaping rewards for approach sector position and glideslope
        if self._sim_parameters.reward_shaping:
            reward += self._reward_approach_position(self._d_faf, self._runway.phi_to_runway, self._phi_rel_faf)
            reward += self._reward_approach_angle(self._d_faf, self._runway.phi_to_runway,
                                                  self._phi_rel_faf, self._airplane.phi)
            reward += self._reward_glideslope(self._d_faf, self._airplane.h,
                                              self._runway.phi_to_runway, self._phi_rel_faf)

        if self._sim_parameters.normalize_state:
            state = (state - self.normalization_state_min - 0.5 * self.normalization_state_max) \
                    / (0.5 * self.normalization_state_max)

        self.last_reward = reward
        self.total_reward += reward
        return state, reward, self.done, {}

    def _reward_glideslope(self, d_faf, h, phi_to_runway, phi_rel_to_faf):
        """
        Calculates a reward based upon the on glideslope performance of the airplane, weighted with the position
        relative to the FAF.

        :param d_faf: Distance to the faf, in nm
        :param h: Altitude of the airplane
        :param phi_to_runway: Approach course/Runway heading
        :param phi_rel_to_faf: Angle of the airplane to the FAF
        :return: Reward factor
        """
        on_gp_altitude = np.tan(np.radians(3)) * d_faf * model.nautical_miles_to_feet + self._faf_mva
        position_factor = self._reward_approach_position(d_faf, phi_to_runway, phi_rel_to_faf, 0.4)
        altitude_diff_factor = 1 - np.abs(h - on_gp_altitude) / self._airplane.h_max
        return altitude_diff_factor * position_factor * 1.2

    def _reward_approach_position(self, d_faf, phi_to_runway, phi_rel_to_faf, faf_power=0.2):
        """
        Provides a reward based upon the position of the aircraft in relation to the final approach course

        The closer the plane is to the FAF, along the approach course, of the side away from the runway the higher
        the reward given by this function.

        :param d_faf: Distance to the faf, in nm
        :param phi_to_runway: Approach course/Runway heading
        :param phi_rel_to_faf: Angle of the airplane to the FAF
        :param faf_power: Determines the fall-off based upon the distance. Use larger values for rewards only close
        to the FAF
        :return: Reward factor
        """
        # advanced award for approach sector location
        reward_faf = 1 / np.maximum(np.power(d_faf, faf_power), 1)
        reward_app_angle = np.power(np.abs(model.relative_angle(phi_to_runway, phi_rel_to_faf)) / 180, 1.5)
        return reward_faf * reward_app_angle * 0.8

    def _reward_approach_angle(self, d_faf, phi_to_runway, phi_rel_to_faf, phi_plane):
        """
        Provides a reward based upon the angle of the aircraft relative to an intercept of the approach course.

        Weighted by the distance to the FAF.

        :param d_faf: Distance to the faf, in nm
        :param phi_to_runway: Approach course/Runway heading
        :param phi_rel_to_faf: Angle of the airplane to the FAF
        :param phi_plane: Heading of the plane
        :return: Reward factor
        """

        def reward_model(angle):
            return np.power(-np.power((angle - 22.5) / 202, 2) + 1, 32)

        plane_to_runway = model.relative_angle(phi_to_runway, phi_plane)
        side = np.sign(model.relative_angle(phi_to_runway, phi_rel_to_faf))
        position_factor = self._reward_approach_position(d_faf, phi_to_runway, phi_rel_to_faf, 0.4)

        return reward_model(side * plane_to_runway) * position_factor * 0.8

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
        self._d_faf = np.hypot(to_faf_x, to_faf_y)
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
                return 0.0
                # return 0.0
        except ValueError:
            # invalid action, outside of permissible range
            return -1.0
        return 0.0

    def reset(self):
        """
        Reset the environment

        Creates a new airplane instance
        :return:
        """
        self.done = False
        self._airplane = model.Airplane(self._sim_parameters, "FLT01", 10, 51, 16000, 90, 250)
        self._airplane = model.Airplane(self._sim_parameters, "FLT01", 47.2, 35.1, 2700, 359, 250)
        self.state = self._get_state()
        return self.state

    def render(self, mode='human'):
        """
        Rendering the environments state
        :type mode: Either "human" for direct to screen rendering or "rgb_array"
        """
        if self.viewer is None:
            self._padding = 10
            screen_width = 600

            world_size_x = self._world_x_max - self._world_x_min
            world_size_y = self._world_y_max - self._world_y_min
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
        self._render_reward()

        return self.viewer.render(mode == 'rgb_array')

    def _render_reward(self):
        total_reward = "Total reward: %.2f" % self.total_reward
        last_reward = "Last reward: %.2f" % self.last_reward

        label_total = Label(total_reward, 10, 40, bold=False)
        label_last = Label(last_reward, 10, 25, bold=False)

        self.viewer.add_onetime(label_total)
        self.viewer.add_onetime(label_last)

    def _render_airplane(self, airplane: model.Airplane):
        """
        Renders the airplane symbol and adjacent information onto the screen

        Already supports multiple airplanes in the environment.

        :param airplane: Airplane to render
        :return: None
        """
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
        """
        Render the approach path on the screen

        Currently only supports a single runway and a single approach corridor

        :return: None
        """
        iaf_x = self._runway.corridor.iaf[0][0]
        iaf_y = self._runway.corridor.iaf[1][0]
        dashes = 48
        runway_vector = self._screen_vector(self._runway.x, self._runway.y)
        runway_iaf = np.array([[iaf_x - self._runway.x], [iaf_y - self._runway.y]]) * self._scale
        for i in range(int(dashes / 2 + 1)):
            start = runway_vector + runway_iaf / dashes * 2 * i
            end = runway_vector + runway_iaf / dashes * (2 * i + 1)
            dash = rendering.PolyLine([start, end], False)
            dash.set_color(*ColorScheme.lines_info)
            self.viewer.add_geom(dash)

    def _render_faf(self):
        """
        Renders the final approach fix symbol onto the screen

        Currently only supports a single runway and a single approach corridor

        :return: None
        """
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
        """
        Renders the outlines of the minimum vectoring altitudes onto the screen.

        :return: None
        """

        def transform_world_to_screen(coords):
            return [((coord[0] - self._world_x_min) * self._scale + self._padding,
                     (coord[1] - self._world_y_min) * self._scale + self._padding) for coord in coords]

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
        """
        Renders the runway symbol onto the screen

        Currently only supports a single runway and a single approach corridor
        :return: None
        """
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
        """
        Converts an in world vector to an on screen vector by shifting and scaling
        :param x: World vector x
        :param y: World vector y
        :return: Numpy array vector with on screen coordinates
        """
        return np.array([
            [(x - self._world_x_min) * self._scale],
            [(y - self._world_y_min) * self._scale]
        ])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
