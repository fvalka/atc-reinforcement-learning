import pyglet
from gym.envs.classic_control.rendering import Geom


class Text(Geom):
    def __init__(self, text, x, y):
        super().__init__()
        self.text = text
        self.x = x
        self.y = y

    def render1(self):
        label = pyglet.text.Label(self.text,
                                  font_name='Arial',
                                  font_size=12,
                                  x=self.x, y=self.y,
                                  color=(0, 0, 0, 255))
        label.draw()
