import math
import random
from typing import List

import numpy as np
import shapely.geometry as geom
import shapely.ops
from numba import jit

nautical_miles_to_feet = 6076  # ft/nm


class Airplane:
    def __init__(self, sim_parameters, name, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300):
        """
        State of one aircraft simulated in the environment

        :param sim_parameters: Definition of the simulation, timestep and more
        :param name: Name of the flight/airplane
        :param x: Position in cartesian world coordinates
        :param y: Position in cartesian world coordinates
        :param h: Height [feet]
        :param phi: Angle of direction, between 1 and 360 degrees
        :param v: Speed [knots]
        :param v_min: Min. speed [knots]
        :param v_max: Max. speed [knots]
        :param h_min: Min. speed [feet]
        :param h_max: Max. speed [feet]        
        """
        self.sim_parameters = sim_parameters
        self.name = name
        self.x = x
        self.y = y
        self.h = h
        if (h < h_min) or (h > h_max):
            raise ValueError("invalid altitude")
        self.v = v
        if (v < v_min) or (v > v_max):
            raise ValueError("invalid velocity")
        self.phi = phi
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max
        self.h_dot_min = -41
        self.h_dot_max = 15
        self.a_max = 5
        self.a_min = -5
        self.phi_dot_max = 3
        self.phi_dot_min = -3
        self.position_history = []
        self.id = random.randint(0, 32767)

    def above_mva(self, mvas):
        for mva in mvas:
            if mva.area.contains(geom.Point(self.x, self.y)):
                return self.h >= mva.height
        raise ValueError('Outside of airspace')

    def action_v(self, action_v):
        """
        Updates the aircrafts state to a new target speed.

        The target speed will be bound by [v_min, v_max] and the rate of change by [a_min, a_max]

        :param action_v: New target speed of the aircraft
        :return: Change has been made to the self speed
        """
        if action_v < self.v_min:
            raise ValueError("invalid speed")
        if action_v > self.v_max:
            raise ValueError("invalid speed")
        delta_v = action_v - self.v
        # restrict to max acceleration, upper bound
        delta_v = min(delta_v, self.a_max * self.sim_parameters.timestep)

        # restrict to min acceleration, lower bound
        delta_v = max(delta_v, self.a_min * self.sim_parameters.timestep)

        self.v = self.v + delta_v

    def action_h(self, action_h):
        """
        Updates the aircrafts state to a new target height.

        The target height will be bound by [h_min, h_max] and the climb/descend rate by [h_dot_min, h__dot_max]

        :param action_h: New target height of the aircraft
        :return: Change has been made to the height
        """
        if action_h < self.h_min:
            raise ValueError("invalid altitude")
        if action_h > self.h_max:
            raise ValueError("invalid altitude")
        delta_h = action_h - self.h
        # restrict to max climb speed, upper bound
        delta_h = min(delta_h, self.h_dot_max * self.sim_parameters.timestep)

        # restrict to max decend speed, lower bound
        delta_h = max(delta_h, self.h_dot_min * self.sim_parameters.timestep)

        self.h = self.h + delta_h

    def action_phi(self, action_phi):
        """
        Updates the aircrafts state to a new course.

        The target course will be bound by [phi_dot_min, phi_dot_max]

        :param action_phi: New target course of the aircraft
        :return: Change has been made to the target heading
        """
        delta_phi = action_phi - self.phi
        # restrict to max climb speed, upper bound
        delta_phi = min(delta_phi, self.phi_dot_max * self.sim_parameters.timestep)

        # restrict to max decend speed, lower bound
        delta_phi = max(delta_phi, self.phi_dot_min * self.sim_parameters.timestep)

        self.phi = self.phi + delta_phi

    def step(self):
        self.position_history.append((self.x, self.y))

        # convert speed vector to nautical miles per second
        v_unrotated = np.array([[0], [(self.v / 3600) * self.sim_parameters.timestep]])
        delta_x_y = np.dot(rot_matrix(self.phi), v_unrotated)
        self.x += delta_x_y[0][0]
        self.y += delta_x_y[1][0]


class SimParameters:
    def __init__(self, timestep: float, precision: float = 0.5, reward_shaping: bool = True,
                 normalize_state: bool = True, discrete_action_space: bool = False):
        """
        Defines the simulation parameters

        :param timestep: Timestep size [seconds]
        :param precision: Epsilon for 0 comparisons
        """
        self.timestep = timestep
        self.precision = precision
        self.reward_shaping = reward_shaping
        self.normalize_state = normalize_state
        self.discrete_action_space = discrete_action_space


class Corridor:
    x: int
    y: int
    h: int
    phi_from_runway: int
    phi_to_runway: int

    def __init__(self, x: int, y: int, h: int, phi_from_runway: int):
        """
        Defines the corridor that belongs to a runway
        """

        self.x = x
        self.y = y
        self.h = h
        self.phi_from_runway = phi_from_runway
        self.phi_to_runway = (phi_from_runway + 180) % 360
        self._faf_iaf_normal = np.dot(rot_matrix(self.phi_from_runway), np.array([[0], [1]]))

        faf_threshold_distance = 7.4
        faf_angle = 45
        self.faf_angle = faf_angle
        faf_iaf_distance = 3
        faf_iaf_distance_corner = faf_iaf_distance / math.cos(math.radians(faf_angle))
        self.faf = np.array([[x], [y]]) + np.dot(rot_matrix(phi_from_runway),
                                                 np.array([[0], [faf_threshold_distance]]))
        self.corner1 = np.dot(rot_matrix(faf_angle),
                              np.dot(rot_matrix(phi_from_runway), [[0], [faf_iaf_distance_corner]])) + self.faf
        self.corner2 = np.dot(rot_matrix(-faf_angle),
                              np.dot(rot_matrix(phi_from_runway), [[0], [faf_iaf_distance_corner]])) + self.faf
        self.corridor_horizontal = geom.Polygon([self.faf, self.corner1, self.corner2])
        self.iaf = np.array([[x], [y]]) + np.dot(rot_matrix(phi_from_runway),
                                                 np.array([[0], [faf_threshold_distance + faf_iaf_distance]]))
        self.corridor1 = geom.Polygon([self.faf, self.corner1, self.iaf])
        self.corridor2 = geom.Polygon([self.faf, self.corner2, self.iaf])

        self.corridor_horizontal_list = np.array(list(self.corridor_horizontal.exterior.coords))
        self.corridor1_list = np.array(list(self.corridor1.exterior.coords))
        self.corridor2_list = np.array(list(self.corridor2.exterior.coords))

    def inside_corridor(self, x, y, h, phi):
        """
        Performance optimized inside corridor check.

        :param x: x-position of the airplane [nm]
        :param y: y-position of the airplane [nm]
        :param h: Altitude of the airplane [ft]
        :param phi: Heading [deg]
        :return: Whether the airplane is inside the approach corridor
        """
        if not ray_tracing(x, y, self.corridor_horizontal_list):
            return False

        p = np.array([[x, y]])
        t = np.dot(p - np.transpose(self.faf), self._faf_iaf_normal)
        projection_on_faf_iaf = self.faf + t * self._faf_iaf_normal
        h_max_on_projection = np.linalg.norm(projection_on_faf_iaf - np.array([[self.x], [self.y]])) * \
                              math.tan(3 * math.pi / 180) * nautical_miles_to_feet + self.h

        if not h <= h_max_on_projection:
            return False

        return self._inside_corridor_angle(x, y, phi)

    def _inside_corridor_angle(self, x, y, phi):

        direction_correct = False

        to_runway = self.phi_to_runway
        beta = self.faf_angle - np.arccos(
            np.dot(
                np.transpose(np.dot(rot_matrix(to_runway), np.array([[0], [1]]))),
                np.dot(rot_matrix(phi), np.array([[0], [1]]))
            )
        )[0][0]
        min_angle = self.faf_angle - beta
        if ray_tracing(x, y, self.corridor1_list) and min_angle <= relative_angle(to_runway,
                                                                                  phi) <= self.faf_angle:
            direction_correct = True
        elif ray_tracing(x, y, self.corridor2_list) and min_angle <= relative_angle(phi,
                                                                                    to_runway) <= self.faf_angle:
            direction_correct = True

        return direction_correct


class Runway:
    corridor: Corridor

    def __init__(self, x, y, h, phi):
        """
        Defines position and orientation of the runway
        """
        self.x = x
        self.y = y
        self.h = h
        self.phi_from_runway = phi
        self.phi_to_runway = (phi + 180) % 360
        self.corridor = Corridor(x, y, h, phi)

    def inside_corridor(self, x: int, y: int, h: int, phi: int):
        """
        Checks if an airplane at a specific 3D point and heading is inside the approach corridor

        :param x: X position of the airplane
        :param y: Y position of the airplane
        :param h: Altitude of the airplane
        :param phi: Heading of the airplane [degrees]
        """
        return self.corridor.inside_corridor(x, y, h, phi)


class MinimumVectoringAltitude:
    area: geom.Polygon
    height: int

    def __init__(self, area: geom.Polygon, height: int):
        self.area = area
        self.height = height
        self.area_as_list = np.array(list(area.exterior.coords))
        self.outer_bounds = area.bounds


class Airspace:
    mvas: List[MinimumVectoringAltitude]

    def __init__(self, mvas: List[MinimumVectoringAltitude], runway: Runway):
        """
        Defines the airspace. Each area is a polygon entered as a list of tuples, Pass several areas as a list or tuple
        MVA is defined by a number (height in feet), pass as a list or tuple equal to the number of
        """
        self.mvas = mvas
        self.runway = runway

    def find_mva(self, x, y):
        for mva in self.mvas:
            bounds = mva.outer_bounds
            # tuple with minx, miny, maxx, maxy
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                if ray_tracing(x, y, mva.area_as_list):
                    return mva
        raise ValueError('Outside of airspace')

    def get_mva_height(self, x, y):
        return self.find_mva(x, y).height

    def get_bounding_box(self):
        """
        Returns the bounding box of the airspace

        :return Tuple with minx, miny, maxx, maxy
        """
        combined_poly = self.get_outline_polygon()
        return combined_poly.bounds

    def get_outline_polygon(self):
        polys: List[geom.polygon] = [mva.area for mva in self.mvas]
        combined_poly = shapely.ops.unary_union(polys)
        return combined_poly


class EntryPoint:

    def __init__(self, x: float, y: float, phi: int, levels: List[int]):
        self.x = x
        self.y = y
        self.phi = phi
        self.levels = levels


@jit(nopython=True)
def ray_tracing(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True)
def relative_angle(angle1, angle2):
    return (angle2 - angle1 + 180) % 360 - 180


@jit(nopython=True)
def rot_matrix(phi):
    phi = math.radians(phi)
    return np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
