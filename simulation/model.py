import math
import shapely.geometry as shape
from typing import List
import numpy as np

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
        self.phi_dot_max = 3
        self.phi_dot_min = 3

    def above_mva(self, mvas):
        for mva in mvas:
            if mva.area.contains(shape.Point(self.x, self.y)):
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

        return math.abs(delta_v) >= self.sim_parameters.precision

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
    def __init__(self, timestep, precision=0.0001):
        """
        Defines the simulation parameters

        :param timestep: Timestep size [seconds]
        :param precision: Epsilon for 0 comparisons
        """
        self.timestep = timestep
        self.precision = precision


class MinimumVectoringAltitude:
    def __init__(self, area, height):
        self.area = area
        self.height = height


class Airspace:
    mvas: List[MinimumVectoringAltitude]

    def __init__(self, mvas: List[MinimumVectoringAltitude]):
        """
        Defines the airspace. Each area is a polygon entered as a list of touples, Pass several areas as a list or touple
        MVA is defined by a number (heigt in feet), pass as a list or touple equal to the number of 
        """
        self.mvas = mvas
        
    def find_mva(self,x,y):
       for mva in self.mvas:
            if mva.area.contains(shape.Point(x,y)):
                return mva
       raise ValueError('Outside of airspace')
       
    def get_mva(self,x,y):
        self.find_mva(x,y).height

class Runway:
    def __init__(self,x,y,h,phi,airspace):
        """
        Defines position and orientation of the runway
        """
        self.x = x
        self.y = y
        self.airspace = airspace
        self.h = h
        self.phi = phi
        airspace.find_mva(self.x,self.y)
    
class Corridor:
    def __init__(self,runway):
        """
        Defines the corridor that belongs to a runway
        """
        def rot_matrix(phi):
            phi = math.radians(phi)
            return np.array([[math.cos(phi),math.sin(phi)],[-math.sin(phi),math.cos(phi)]]) 
        self.runway = runway
        faf_distance = 8
        faf_angle = 45
        faf_iaf_distance = 3
        faf_iaf_distance_corner = faf_iaf_distance/math.cos(math.radians(faf_angle))
        self.faf = np.array([[runway.x],[runway.y]]) + np.dot(rot_matrix(runway.phi),np.array([[0],[faf_distance]]))
        self.corner1 = np.dot(rot_matrix(faf_angle),np.dot(rot_matrix(runway.phi),[[0],[faf_iaf_distance_corner]]))+self.faf
        self.corner2 = np.dot(rot_matrix(-faf_angle),np.dot(rot_matrix(runway.phi),[[0],[faf_iaf_distance_corner]]))+self.faf
        self.corridor_h = shape.Polygon([self.faf,self.corner1,self.corner2])
        
class Object():
    pass

runway = Object()
runway.x = 5
runway.y = 5
runway.phi = 270