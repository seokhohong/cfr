from enum import Enum

class PokerActions(Enum):
    RAISE_1 = 'RAISE 1x',
    RAISE_2 = 'RAISE 2x',
    RAISE_3 = 'RAISE 3x',
    ALL_IN = 'ALL_IN',
    CHECK = 'CHECK',
    FOLD = 'FOLD',
    CALL = 'CALL'

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()