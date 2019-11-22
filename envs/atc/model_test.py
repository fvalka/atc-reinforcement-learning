import unittest
from typing import List

import shapely.geometry as shape
from . import model


class ModelTestCase(unittest.TestCase):

    def test_airspace_get_mvas(self):
        x = 34
        y = 1
        test_mvas = self.generate_mvas()
        test_airspace = self.generate_airspace(test_mvas, self.generate_runway())
        z = test_airspace.get_mva_height(x, y)
        self.assertEqual(z, 3500)

    def test_inside_corridor_true_when_on_correct_side(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        mva = test_airspace.get_mva_height(test_runway.corridor.faf[0][0], test_runway.corridor.faf[1][0])
        x = 19
        y = 10
        h = mva + 300
        phi = 30
        self.assertEqual(test_runway.inside_corridor(x, y, h, phi), True)

    def test_inside_corridor_false_when_right_position_with_wrong_heading(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        mva = test_airspace.get_mva_height(test_runway.corridor.faf[0][0], test_runway.corridor.faf[1][0])
        x = 19
        y = 10
        h = mva
        phi = 330
        self.assertEqual(test_runway.inside_corridor(x, y, h, phi), False)


    def test_inside_corridor_false_when_on_wrong_side(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        x = 21
        y = 10
        h = 0
        phi = 30
        self.assertEqual(test_runway.corridor._inside_corridor_angle(x, y, phi), False)

    def test_inside_corridor_false_when_on_wrong_side_other_side(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        x = 19
        y = 10
        h = 0
        phi = 340
        self.assertEqual(test_runway.corridor._inside_corridor_angle(x, y, phi), False)

    def test_inside_corridor_false_when_on_opposite_direction(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        x = 19
        y = 10
        h = 0
        phi = 190
        self.assertEqual(test_runway.corridor._inside_corridor_angle(x, y, phi), False)

    def test_inside_corridor_true_when_on_other_correct_side(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)
        x = 21
        y = 10
        h = 0
        phi = 340
        self.assertEqual(test_runway.corridor._inside_corridor_angle(x, y, phi), True)

    def test_bounding_box_airspace_multiple_mva(self):
        test_mvas = self.generate_mvas()
        test_runway = self.generate_runway()
        test_airspace = self.generate_airspace(test_mvas, test_runway)

        # min_x, min_y, max_x, max_y
        result = test_airspace.get_bounding_box()

        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[2], 35.0)
        self.assertEqual(result[3], 40.0)

    def generate_mvas(self):
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]
        return mvas

    def generate_airspace(self, mvas: List[model.MinimumVectoringAltitude], runway: model.Runway):
        airspace = model.Airspace(mvas, runway)
        return airspace

    def generate_runway(self):
        x = 20
        y = 20
        h = 0
        phi = 180
        runway = model.Runway(x, y, h, phi)
        return runway


if __name__ == '__main__':
    unittest.main()
