import shapely.geometry as shape
from typing import List

from . import model


class Scenario:
    runway: model.Runway
    mvas: List[model.MinimumVectoringAltitude]
    airspace: model.Airspace
    entrypoints: List[model.EntryPoint]


class SimpleScenario(Scenario):

    def __init__(self, random_entrypoints=False):
        mva_1 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 0), (35, 26)]), 3500)
        mva_2 = model.MinimumVectoringAltitude(shape.Polygon([(15, 0), (35, 26), (35, 30), (15, 30), (15, 27.8)]), 2400)
        mva_3 = model.MinimumVectoringAltitude(shape.Polygon([(15, 30), (35, 30), (35, 40), (15, 40)]), 4000)
        mva_4 = model.MinimumVectoringAltitude(shape.Polygon([(0, 10), (15, 0), (15, 28.7), (0, 17)]), 8000)
        mva_5 = model.MinimumVectoringAltitude(shape.Polygon([(0, 17), (15, 28.7), (15, 40), (0, 32)]), 6500)
        self.mvas = [mva_1, mva_2, mva_3, mva_4, mva_5]

        x = 20
        y = 20
        h = 0
        phi = 130
        self.runway = model.Runway(x, y, h, phi)
        self.airspace = model.Airspace(self.mvas, self.runway)
        self.entrypoints = [
            model.EntryPoint(5, 35, 90, [150])
        ]


class LOWW(Scenario):
    def __init__(self, random_entrypoints=False):
        super().__init__()

        self.mvas = [
            model.MinimumVectoringAltitude(shape.Polygon([
                (48.43, 2.09),
                (39.36, 4.22),
                (27.26, 20.01),
                (54.03, 12.95),
                (48.43, 2.09)
            ]), 4800),
            model.MinimumVectoringAltitude(shape.Polygon([
                (27.26, 20.01),
                (26.37, 21.35),
                (29.73, 26.39),
                (28.83, 31.09),
                (34.32, 25.55),
                (46.08, 22.36),
                (42.47, 16),
                (27.26, 20.01)
            ]), 3700),
            model.MinimumVectoringAltitude(shape.Polygon([
                (26.37, 21.35),
                (13.15, 38.60),
                (22.0, 36.13),
                (22.0, 30.65),
                (29.73, 26.39),
                (26.37, 21.35)
            ]), 5700),
            model.MinimumVectoringAltitude(shape.Polygon([
                (29.73, 26.39),
                (22.0, 30.65),
                (22.0, 36.13),
                (13.15, 38.60),
                (8, 45.68),
                (18.75, 44.98),
                (28.83, 31.09),
                (29.73, 26.39)
            ]), 4600),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (18.75, 44.98),
                (22.0, 45.68),
                (26.37, 43.08),
                (28.83, 31.09)
            ]), 4100),
            model.MinimumVectoringAltitude(shape.Polygon([
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (28.83, 31.09)
            ]), 4000),
            model.MinimumVectoringAltitude(shape.Polygon([
                (22.0, 45.68),
                (18.75, 44.98),
                (8, 45.68),
                (4.08, 50.25),
                (15.73, 76.12),
                (29.73, 80.71),
                (56.16, 82.05),
                (58.51, 69.84),
                (42.94, 71.97),
                (22.56, 65.36),
                (16.17, 50.25),
                (23.23, 49.01),
                (22.0, 45.68)
            ]), 3500),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (34.32, 25.55),
                (31.5, 28.4),
                (36.22, 35.46),
                (44.46, 31.76),
                (46.08, 22.36)
            ]), 3000),
            model.MinimumVectoringAltitude(shape.Polygon([
                (31.5, 28.4),
                (28.83, 31.09),
                (28.83, 33.45),
                (31.29, 34.12),
                (29.73, 41.29),
                (26.9, 40.47),
                (26.37, 43.08),
                (22.0, 45.68),
                (23.23, 49.01),
                (31.29, 48.01),
                (30.17, 45.71),
                (32.19, 44.98),
                (35.14, 41.62),
                (36.22, 42.29),
                (37.56, 36.69),
                (36.22, 35.46),
                (31.5, 28.4)
            ]), 3500),
            model.MinimumVectoringAltitude(shape.Polygon([
                (35.14, 41.62),
                (32.19, 44.98),
                (30.17, 45.71),
                (31.29, 48.01),
                (23.23, 49.01),
                (16.17, 50.25),
                (22.56, 65.36),
                (36.58, 69.91),
                (39.47, 60.55),
                (35.73, 59.13),
                (36.22, 56.18),
                (38.46, 53.72),
                (34.32, 45.68),
                (35.14, 41.62)
            ]), 3200),
            model.MinimumVectoringAltitude(shape.Polygon([
                (46.08, 22.36),
                (44.95, 28.91),
                (53.5, 31.43),
                (57.97, 41.89),
                (47.17, 55.97),
                (40.75, 53.72),
                (38.46, 53.72),
                (36.22, 56.18),
                (35.73, 59.13),
                (39.47, 60.55),
                (36.58, 69.91),
                (42.94, 71.97),
                (58.51, 69.84),
                (54.78, 60.01),
                (68.15, 38.6),
                (66.34, 36.85),
                (65.53, 30.62),
                (62.92, 29.97),
                (66.58, 20.58),
                (52.88, 18.68),
                (51.64, 21.35),
                (46.08, 22.36)
            ]), 2700),
            model.MinimumVectoringAltitude(shape.Polygon([
                (44.95, 28.91),
                (44.46, 31.76),
                (36.22, 35.46),
                (37.56, 36.69),
                (36.22, 42.29),
                (35.14, 41.62),
                (34.32, 45.68),
                (38.46, 53.72),
                (40.75, 53.72),
                (47.17, 55.97),
                (57.97, 41.89),
                (53.5, 31.43),
                (44.95, 28.91)
            ]), 2600)]

        self.runway = model.Runway(45.16, 43.26, 586, 160)

        self.airspace = model.Airspace(self.mvas, self.runway)

        if random_entrypoints:
            self.entrypoints = [
                model.EntryPoint(10, 51, 90, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(17, 74.6, 120, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(19.0, 34.0, 45, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(29.8, 79.4, 170, [130, 150, 170, 190, 210, 230]),
                model.EntryPoint(54.0, 80.5, 230, [140, 160, 180, 200, 220, 240]),
                model.EntryPoint(53.0, 60.0, 260, [140, 160, 180, 200, 220, 240]),
                model.EntryPoint(66.0, 39.0, 290, [140, 160, 180, 200, 220]),
                model.EntryPoint(64.4, 22.0, 320, [140, 160, 180, 200, 220]),
                model.EntryPoint(46.0, 7.0, 320, [140, 160, 180, 200, 220, 240, 260])
            ]
        else:
            self.entrypoints = [
                model.EntryPoint(10, 51, 90, [150])
            ]
