

def create_player_set(num):
    players = [Player(i) for i in range(num)]
    players[-1].set_next(players[0])
    for i in range(num - 1):
        players[i].set_next(players[i + 1])

    return players

class Player:
    def __init__(self, index):
        self._name = str(index)
        self._index = index

    def set_next(self, next_player):
        self.next_player = next_player

    def get_next(self):
        return self.next_player

    def get_index(self):
        return self._index

    def __repr__(self):
        return self._name

    def __str__(self):
        return self.__repr__()

ChancePlayer = Player('CHANCE')