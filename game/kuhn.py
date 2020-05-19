from game.poker import PokerActions
from game.player import ChancePlayer
import itertools
import random
import numpy as np

class GameStateBase:

    def __init__(self, parent, player_to_move, actions):
        self.parent = parent
        self._player_to_move = player_to_move
        self._children = None
        self.actions = actions

    def get_children(self):
        if not self._children:
            self._create_children()

        return self._children

    def _create_children(self):
        raise NotImplementedError("Abstract Method")

    def play(self, action):
        if not self._children:
            self._create_children()

        return self._children[action]

    def is_chance(self):
        return self._player_to_move == ChancePlayer

    def get_player_to_move(self):
        return self._player_to_move

    def inf_set(self):
        raise NotImplementedError("Please implement information_set method")

class KuhnGame:
    def __init__(self, players, card_set, num_deal):
        self._players = players
        self._cards = card_set
        self._num_deal = num_deal

    def create_root_node(self):
        possible_hands = list(self.enumerate_possible_hands())
        children = {
            cards: KuhnPlayerMoveGameState(
                self, self._players, self._players[0], [],  cards, [PokerActions.RAISE_1, PokerActions.CHECK]
            ) for cards in possible_hands
        }
        return ChanceGameState(children, possible_hands)

    def enumerate_possible_hands(self):
        return itertools.permutations(self._cards, len(self._players) * self._num_deal)

    def get_players(self):
        return self._players

class ChanceGameState(GameStateBase):
    def __init__(self, children, actions):
        super().__init__(self, player_to_move=ChancePlayer, actions=actions)
        self._children = children
        self._chance_prob = 1. / len(self._children)

    def is_terminal(self):
        return False

    def inf_set(self):
        return "."

    def chance_prob(self):
        return self._chance_prob

    def sample_one(self):
        return random.choice(list(self.get_children().values()))


class KuhnPlayerMoveGameState(GameStateBase):

    def __init__(self, parent, players, player_to_move, actions_history, cards, actions):
        super().__init__(parent=parent, player_to_move=player_to_move, actions=actions)

        self.actions_history = actions_history
        self.cards = cards
        self._children = None
        self._players = players


        known_card = self.cards[self.get_player_to_move().get_index()]
        action_list = ".".join([str(a) for _, a in self.actions_history])
        self._information_set = "{0}.{1}".format(known_card, action_list)

    def _create_children(self):
        next_player = self.get_player_to_move().get_next()
        self._children = {
            a : KuhnPlayerMoveGameState(
                self,
                self._players,
                next_player,
                self.actions_history + [(self.get_player_to_move(), a)],
                self.cards,
                self.__get_actions_in_next_round(a)
            ) for a in self.actions
        }

    def __get_actions_in_next_round(self, a):
        if self.__action_has_been_played(PokerActions.RAISE_1) or a == PokerActions.RAISE_1:
            if self.__player_who_played(PokerActions.RAISE_1) != self.get_player_to_move().get_next():
                return [PokerActions.FOLD, PokerActions.CALL]
            return []
        elif not self.__action_has_been_played(PokerActions.RAISE_1) and a != PokerActions.RAISE_1:
            # all checks
            if len(self.actions_history) + 1 == len(self._players):
                return []
            return [PokerActions.RAISE_1, PokerActions.CHECK]
        else:
            return []

    def __action_has_been_played(self, a):
        for player, action in self.actions_history:
            if action == a:
                return True
        return False

    def __player_who_played(self, a):
        for player, action in self.actions_history:
            if action == a:
                return player
        return None

    def inf_set(self):
        return self._information_set

    def is_terminal(self):
        return self.actions == []

    def pot_size(self):
        pot = len(self._players)
        for player, action in self.actions_history:
            if action == PokerActions.RAISE_1 or action == PokerActions.CALL:
                pot += 1
        return pot

    def pot_contribution(self, which_player):
        for player, action in self.actions_history:
            if player == which_player:
                if action == PokerActions.RAISE_1 or action == PokerActions.CALL:
                    return 2
        return 1

    def evaluation(self):
        if not self.is_terminal():
            raise RuntimeError("trying to evaluate non-terminal node")

        result_vector = np.repeat(-1, len(self._players))

        pot = self.pot_size()

        active_players = self._players.copy()
        for player, action in self.actions_history:
            if action == PokerActions.FOLD:
                active_players.remove(player)

        winner_index = np.argmax([self.cards[player.get_index()] for player in active_players])
        winner_player_index = active_players[int(winner_index)]

        for player in self._players:
            if player == winner_player_index:
                result_vector[player.get_index()] = pot - self.pot_contribution(player)
            else:
                result_vector[player.get_index()] = - self.pot_contribution(player)

        return result_vector