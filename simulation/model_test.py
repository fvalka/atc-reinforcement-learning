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
        
    def test_inside_corrodidor(self): 
        test_mvas = self.generate_mvas()
        test_airspace = self.generate_airspace(test_mvas)
        test_runway = self.generate_runway(test_airspace)
        test_corridor = self.generate_corridor(test_runway)
        x = 19
        y = 10
        h = 0
        phi = 30
        zz = test_corridor._inside_corridor_angle(x, y, h, phi)
        self.assertEqual(zz, True)

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

    def generate_runway(self, airspace):
        x = 20
        y = 20
        h = 0
        phi = 180        
        runway = model.Runway(x, y, h, phi, airspace)
        return runway
    
    def generate_corridor(self, runway):
        corridor = model.Corridor(runway)
        return corridor

if __name__ == '__main__':
    unittest.main()
