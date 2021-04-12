from enum import Enum


class PoliticalSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PoliticalBias:
    def __init__(self, side: PoliticalSide, probability: float):
        self.side = side
        self.probability = probability
