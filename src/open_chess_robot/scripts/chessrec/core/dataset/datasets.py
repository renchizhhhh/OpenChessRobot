from enum import Enum


class Datasets(Enum):
    """Enumeration of the dataset split.
    """
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HIGH = "high"
    LOW = "low"
    LEFT = "left"
    RIGHT = "right"
    DEFAULT = "default"
    ROTATED = "rotated"
    SHIFTED = "shifted"
    GAME = "game"
