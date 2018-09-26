import math

class Airplane:
    _a_min = -5  # kts / sec
    _a_max = 5  # kts / sec

    def __init__(self, sim_parameters, x, y, h, phi, v):
        """
        State of one aircraft simulated in the environment

        :param sim_parameters: Definition of the simulation, timestep and more
        :param x: Position in cartesian world coordinates
        :param y: Position in cartesian world cooridantes
        :param h: Height [feet]
        :param phi: Angle of direction, between 1 and 360 degrees
        :param v: Speed [knots]
        """
        self.sim_parameters = sim_parameters
        self.x = x
        self.y = y
        self.h = h
        self.v = v
        self.phi = phi

    def action_v(self, action_v):
        """
        Updates the aircrafts state to a new target speed.

        The target speed will be bound by [_a_min, _a_max]

        :param action_v: New target speed of the aircraft
        :return: Change has been made to the self speed
        """
        delta_v = action_v - self.v
        # restrict to max acceleration, upper bound
        delta_v = min(delta_v, self._a_max * self.sim_parameters.timestep)

        # restrict to min acceleration, lower bound
        delta_v = max(delta_v, self._a_min * self.sim_parameters.timestep)

        self.v = self.v + delta_v

        return math.abs(delta_v) >= self.sim_parameters.precision


class SimParameters:
    def __init__(self, timestep, precision = 0.0001):
        """
        Defines the simulation parameters

        :param timestep: Timestep size [seconds]
        :param precision: Epsilon for 0 comparisons
        """
        self.timestep = timestep
        self.precision = precision
