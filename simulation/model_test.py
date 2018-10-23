import unittest
import shapely.geometry as shape
import simulation.model as model


class MyTestCase(unittest.TestCase):

    def test_airspace_get_mvas(self):
        x = 34
        y = 1
        test_mvas = self.generate_mvas()
        test_airspace = self.generate_airspace(test_mvas)
        z = test_airspace.get_mva(x, y)
        self.assertEqual(z, 3500)

    def generate_mvas(self):
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]
        return mvas

    def generate_airspace(self, mvas):
        airspace = model.Airspace(mvas)
        return airspace


if __name__ == '__main__':
    unittest.main()
