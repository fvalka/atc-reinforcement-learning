import math
import shapely.geometry as shape

class Airplane:
    def __init__(self, sim_parameters, x, y, h, phi, v, h_min=0, h_max=38000, v_min=100, v_max=300):
        """
        State of one aircraft simulated in the environment

        :param sim_parameters: Definition of the simulation, timestep and more
        :param x: Position in cartesian world coordinates
        :param y: Position in cartesian world cooridantes
        :param h: Height [feet]
        :param phi: Angle of direction, between 1 and 360 degrees
        :param v: Speed [knots]
        :param v_min: Min. speed [knots]
        :param v_max: Max. speed [knots]
        :param h_min: Min. speed [feet]
        :param h_max: Max. speed [feet]        
        """
        self.sim_parameters = sim_parameters
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
        self.h_dot_min = -1000
        self.h_dot_max = 1000
        self.a_max = 5
        self.a_min = -5
        self.phi_dot = [-3,0,3]

    def overMVA(self, MVA):
        if self.h >= MVA:
            return True
        else: return False

    def command(self, h_set=None, v_set=None, phi_set=None):
        self.h_set = h_set
        self.v_set = v_set
        self.phi_set = phi_set
        
    def action_v(self, action_v):
        """
        Updates the aircrafts state to a new target speed.

        The target speed will be bound by [v_min, v_max] and the rate of change by [a_min, a_max]

        :param action_v: New target speed of the aircraft
        :return: Change has been made to the self speed
        """
        if action_v < self.v_min:
            action_v=self.v_min
        if action_v > self.v_max:
            action_v=self.v_max     
        delta_v = action_v - self.v
        # restrict to max acceleration, upper bound
        delta_v = min(delta_v, self.a_max * self.sim_parameters.timestep)

        # restrict to min acceleration, lower bound
        delta_v = max(delta_v, self.a_min * self.sim_parameters.timestep)

        self.v = self.v + delta_v

        return math.abs(delta_v) >= self.sim_parameters.precision

    def action_h(self, action_h):
        """
        Updates the aircrafts state to a new target height.

        The target height will be bound by [h_min, h_max] and the climb/descend rate by [h_dot_min, h__dot_max]

        :param action_h: New target height of the aircraft
        :return: Change has been made to the height
        """
        if action_h < self.h_min:
            action_h=self.h_min
        if action_h > self.h_max:
            action_h=self.h_max           
        delta_h = action_h - self.h
        # restrict to max climb speed, upper bound
        delta_h = min(delta_h, self.h_dot_max * self.sim_parameters.timestep)

        # restrict to max decend speed, lower bound
        delta_h = max(delta_h, self.h_dot_min * self.sim_parameters.timestep)

        self.h = self.h + delta_h

        return math.abs(delta_h) >= self.sim_parameters.precision

    def action_phi(self, action_phi):
        """
        Updates the aircrafts state to a new course.

        The target course will be bound by [phi_dot_min, phi_dot_max]

        :param action_phi: New target course of the aircraft
        :return: Change has been made to the height
        """
        delta_phi = action_phi - self.phi
        # restrict to max climb speed, upper bound
        delta_phi = min(delta_phi, self.phi_dot_max * self.sim_parameters.timestep)

        # restrict to max decend speed, lower bound
        delta_phi = max(delta_phi, self.phi_dot_min * self.sim_parameters.timestep)

        self.phi = self.phi + delta_phi

        return math.abs(delta_phi) >= self.sim_parameters.precision

class SimParameters:
    def __init__(self, timestep, precision = 0.0001):
        """
        Defines the simulation parameters

        :param timestep: Timestep size [seconds]
        :param precision: Epsilon for 0 comparisons
        """
        self.timestep = timestep
        self.precision = precision

class Airspace:
    def __init__(self, area, MVA):
        """
        Defines the airspace. Each area is a polygon entered as a list of touples, Pass several areas as a list or touple
        MVA is defined by a number (heigt in feet), pass as a list or touple equal to the number of 
        """
        if len(area) != len(MVA):
            raise ValueError("Number of areas and MVAs need to match")
        self.areas = []
        for i in area:
            i=shape.Polygon(i)
            self.areas.append(i)
        self.MVAs = []
        for i in MVA:
            self.MVAs.append(i)