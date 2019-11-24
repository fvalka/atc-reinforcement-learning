import pyglet
from gym.envs.classic_control.rendering import Geom

from envs.atc.themes import ColorScheme


class Label(Geom):
    def __init__(self, text, x, y, bold=True):
        super().__init__()
        self.text = text
        self.x = x
        self.y = y
        self.bold = bold

    def render1(self):
        label = pyglet.text.Label(self.text,
                                  font_name='Arial',
                                  font_size=9,
                                  bold=self.bold,
                                  x=self.x, y=self.y,
                                  anchor_x="left", anchor_y="top",
                                  color=ColorScheme.label)
        label.draw()
