from enum import Enum

class Tiles(Enum):

    PLAYER = " () "
    EMPTY = "----"
    NUKE = "NUKE"
    FOOD = "FOOD"
    PLAYER_FOOD = "(FOOD)"
    PLAYER_NUKE = "(NUKE)"

    def __len__(self):
        return 4
