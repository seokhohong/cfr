from game.player import ChancePlayer
import itertools
import random
import numpy as np
from enum import Enum

DIE_SIDES = 6
MAX_DIES = 5

class LDAction:
    def __init__(self, is_call, is_spot_on, die, count):
        self._is_call = is_call
        self._is_spot_on = is_spot_on
        self.die = die
        self.count = count

    def is_a_bet(self):
        return not self.is_call() and not self.is_spot_on()

    def is_call(self):
        return self._is_call

    def is_spot_on(self):
        return self._is_spot_on

    def get_die(self):
        assert not self.is_call() and not self.is_spot_on()
        return self.die

    def get_count(self):
        assert not self.is_call() and not self.is_spot_on()
        return self.count

    def get_bet(self):
        return self.die, self.count

    def __str__(self):
        if self.is_a_bet():
            return "(" + str(self.count) + " " + str(self.die) + "'s)"
        elif self.is_call():
            return 'CALL'
        else:
            return 'SPOT_ON'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, LDAction):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self._is_call == other._is_call and self._is_spot_on == other._is_spot_on \
                    and self.die == other.die and self.count == other.count

    def __hash__(self):
        return hash((self._is_call, self._is_spot_on, self.die, self.count))

CALL = LDAction(True, False, 0, 0)
SPOT_ON = LDAction(False, True, 0, 0)
NO_BET = LDAction(False, False, 0, 0)

def _get_ld_actions(current_bet, max_count):
    current_number, current_quantity = current_bet.get_bet()
    bet_actions = []
    for i in range(max(current_quantity, 1), max_count + 1):
        for j in range(1, DIE_SIDES + 1):
            # restrictive bets
            if (i == current_quantity and j > current_number) or i > current_quantity:
                bet_actions.append(LDAction(False, False, j, i))
    call_actions = [CALL, SPOT_ON] if current_bet != NO_BET else []
    return call_actions + bet_actions

def get_information_set_features(dice, history, num_players, tabular_info=True):
    # dice features
    dice_features = np.zeros(DIE_SIDES)
    for die in dice:
        dice_features[die - 1] += 1

    bet_number = np.array([action.get_bet()[0] for action in history])
    bet_quantity = np.array([action.get_bet()[1] for action in history])
    player_number = np.array([i % num_players for i in range(len(history))])

    feature_set = dice_features, bet_number, bet_quantity, player_number
    if tabular_info:
        return tuple(np.concatenate(feature_set))
    return feature_set

class LDGame:
    def __init__(self, players, num_die):
        self._players = players
        self._num_die = num_die

    def create_root_node(self):
        return RollDieGameState(self._players, self._players[0],
                                self._num_die,
                                dice_states=[], actions=[])

    def get_players(self):
        return self._players

class LDGameStateBase:

    def __init__(self, parent, player_to_move, dice_states, actions):
        self.parent = parent
        self._player_to_move = player_to_move
        self._dice_states = dice_states
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

class RollDieGameState(LDGameStateBase):
    def __init__(self, players, rolling_for_player, dice_per_player, dice_states, actions):
        super().__init__(self, player_to_move=ChancePlayer, dice_states=dice_states, actions=actions)
        self._players = players
        self._rolling_for_player = rolling_for_player
        self._dice_states = dice_states
        self._dice_per_player = dice_per_player

    def _create_children(self):
        next_player = self._rolling_for_player.get_next()

        # not first player's move yet, create more dice roll children
        self._children = {}

        # first player's move
        max_quantity = len(self._players) * self._dice_per_player
        if next_player == self._players[0]:
            actions = _get_ld_actions(NO_BET, max_quantity)
            for dice in self.enumerate_possible_rolls():
                self._children[dice] = LDMoveGameState(self, self._players, next_player,
                                                       [], self._dice_states + [dice], actions=actions)
        if next_player != self._players[0]:
            for dice in self.enumerate_possible_rolls():
                self._children[dice] = RollDieGameState(
                    self._players,
                    next_player,
                    self._dice_per_player,
                    self._dice_states + [dice],
                    actions=[])


    def enumerate_possible_rolls(self):
        return itertools.product(range(1, DIE_SIDES + 1), repeat=self._dice_per_player)

    def is_terminal(self):
        return False

    def inf_set(self):
        return "."

    def chance_prob(self):
        self._chance_prob = 1. / len(self.get_children())
        return self._chance_prob

    def sample_one(self):
        return random.choice(list(self.get_children().values()))

class LDMoveGameState(LDGameStateBase):

    def __init__(self, parent, players, player_to_move, actions_history, dice_states, actions):
        super().__init__(parent=parent, player_to_move=player_to_move,
                         dice_states=dice_states, actions=actions)

        self.actions_history = actions_history
        self._dice_states = dice_states
        self._children = None
        self._players = players
        self._max_bet = len(dice_states) * len(dice_states[0])

        known_dice_states = dice_states[player_to_move.get_index()]

        self._information_set = get_information_set_features(known_dice_states, self.actions_history, len(self._players))

    def _create_children(self):
        next_player = self.get_player_to_move().get_next()
        self._children = {
            a: LDMoveGameState(
                self,
                self._players,
                next_player,
                self.actions_history + [a],
                self._dice_states,
                self._actions_after(a)
            ) for a in self.actions
        }

    def _actions_after(self, action):
        if action.is_call() or action.is_spot_on():
            return []
        return _get_ld_actions(action, self._max_bet)

    def inf_set(self):
        return self._information_set

    def is_terminal(self):
        return self.actions == []

    def play_bet(self, count, die):
        return self.play(LDAction(False, False, die, count))

    def is_ones_valid(self):
        # is first bet a 1?
        return self.actions_history[0].get_die() != 1

    def _raw_number_of_dice(self, value):
        return sum([dice_set.count(value) for dice_set in self._dice_states])

    def _number_of_dice(self, value):
        if self.is_ones_valid():
            return self._raw_number_of_dice(1) + self._raw_number_of_dice(value)
        return self._raw_number_of_dice(value)

    def evaluation(self):
        if not self.is_terminal():
            raise RuntimeError("trying to evaluate non-terminal node")

        result_vector = np.zeros(len(self._players))

        challenged_bet = self.actions_history[-2]
        challenged_player_index = (len(self.actions_history) - 2) % len(self._players)
        challenger_bet = self.actions_history[-1]
        challenger_player_index = (len(self.actions_history) - 1) % len(self._players)

        if challenger_bet.is_call():
            if self._number_of_dice(challenged_bet.get_die()) >= challenged_bet.get_count():
                result_vector[challenger_player_index] = -1
            else:
                result_vector[challenged_player_index] = -1

        if challenged_bet.is_spot_on():
            if self._number_of_dice(challenged_bet.get_die()) == challenged_bet.get_count():
                result_vector[challenger_player_index] = 1
            else:
                result_vector[challenger_player_index] = -1

        # l1 norm
        return result_vector - (sum(result_vector) / len(result_vector))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.actions_history)
