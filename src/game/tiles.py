from enum import Enum

from PySide6.QtGui import QColor


class Tiles(Enum):
    PLAYER = " () "
    EMPTY = "----"
    NUKE = "NUKE"
    FOOD = "FOOD"
    PLAYER_FOOD = "(FOOD)"
    PLAYER_NUKE = "(NUKE)"

    @property
    def colour(self) -> QColor:
        colour_map = {
            Tiles.PLAYER: QColor("blue"),
            Tiles.EMPTY: QColor("white"),
            Tiles.NUKE: QColor("red"),
            Tiles.FOOD: QColor("green"),
            Tiles.PLAYER_FOOD: QColor("cyan"),
            Tiles.PLAYER_NUKE: QColor("magenta")
        }
        return colour_map.get(self, QColor("black"))

    def __len__(self):
        return 4