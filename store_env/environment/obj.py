from minigrid.core.world_object import Box, WorldObj
from minigrid.core.constants import COLORS
from minigrid.utils.rendering import point_in_rect, fill_coords
import numpy as np

from typing import Optional

CUSTOM_COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "orange": np.array([255,140,0])
}

class Shelf(Box):
    def __init__(self, color: str = "purple", contains: Optional[WorldObj] = None):
        super().__init__(color, contains)
        
    def can_pickup(self):
        return False

    def render(self, img):
        # c = COLORS[self.color]
        #
        # # Outline
        # fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        # fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))
        #
        # # Horizontal slit
        # fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def toggle(self, env, pos):
        # # Replace the box by its contents
        # env.grid.set(*pos, self.contains)
        # return True
        return False
