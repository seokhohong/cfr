import numpy as np
from collections import defaultdict
from game.poker import PokerActions
import random

def init_empty_node_maps(players, node, output = None):
    level_1 = lambda: defaultdict(int)
    level_2 = lambda: defaultdict(level_1)
    output = defaultdict(level_2)
    def init_empty_node_maps_recursive(node):
        if node.is_chance():
            for player in players:
                output[player.get_index()][node.inf_set()] = {action: 0. for action in node.actions}
        else:
            player = node.get_player_to_move()
            output[player.get_index()][node.inf_set()] = {action: 0. for action in node.actions}
        for k in node.get_children():
            init_empty_node_maps_recursive(node.get_children()[k])
    #init_empty_node_maps_recursive(node)
    return output

class MultiplayerCFRMBase:
    def __init__(self, root, players, chance_sampling=False):
        self.root = root
        self.cumulative_regrets = init_empty_node_maps(players, root)
        self.cumulative_sigma = init_empty_node_maps(players, root)
        self.nash_equilibrium = init_empty_node_maps(players, root)
        self.chance_sampling = chance_sampling
        self._players = players

class CounterfactualRegretMinimizationBase:

    def __init__(self, root, players, chance_sampling=False):
        self.root = root
        self.cumulative_regrets = init_empty_node_maps(players, root)
        self.cumulative_sigma = init_empty_node_maps(players, root)
        self.nash_equilibrium = init_empty_node_maps(players, root)
        self._learned_strategy = init_empty_node_maps(players, root)
        self.chance_sampling = chance_sampling
        self._players = players

    def get_strategy(self, state):
        player_index = state.get_player_to_move().get_index()
        info_set = state.inf_set()
        actions = state.actions

        normalizing_sum = 0
        strategy = {}
        for action in actions:
            regret = max(self.cumulative_regrets[info_set][action], 0)
            strategy[action] = regret
            normalizing_sum += regret
        for action in actions:
            if normalizing_sum > 0:
                strategy[action] /= normalizing_sum
            else:
                strategy[action] = 1. / len(actions)
        self._learned_strategy[info_set] = strategy
        return strategy


    def compute_nash_equilibrium(self):
        self.__compute_ne_rec(self.root)

    def __compute_ne_rec(self, node):
        if node.is_terminal():
            return
        i = node.inf_set()
        if node.is_chance():
            for player in self._players:
                self.nash_equilibrium[player.get_index()][i] = {a: node.chance_prob() for a in node.actions}
        else:
            player_index = node.get_player_to_move().get_index()
            sigma_sum = sum(self.cumulative_sigma[player_index][i].values())
            self.nash_equilibrium[player_index][i] = {a: self.cumulative_sigma[player_index][i][a] / sigma_sum for a in node.actions}
        # go to subtrees
        for child in node.get_children().values():
            self.__compute_ne_rec(child)

    def _cumulate_sigma(self, player_index, information_set, action, prob):
        #print('Update Sigma', information_set, '          ', action, prob)
        #print('Cum Sigma', information_set, '          ', action, self.cumulative_sigma[information_set][action])
        self.cumulative_sigma[player_index][information_set][action] += prob

    def run(self, iterations):
        raise NotImplementedError("Please implement run method")

    def value_of_the_game(self):
        return self.__value_of_the_game_state_recursive(self.root)

    def repeat_value_for_players(self, value):
        values = {}
        for player in self._players:
            values[player] = value
        return values



    def _cfr_utility_recursive(self, state, reach_vector):
        action_utilities = {}
        node_utilities = np.zeros(len(self._players))
        if state.is_terminal():
            return state.evaluation()
        if state.is_chance():
            if self.chance_sampling:
                # if node is a chance node, lets sample one child node and proceed normally
                return self._cfr_utility_recursive(state.sample_one(), reach_vector)
            else:
                chance_outcomes = {state.play(action) for action in state.actions}
                shared_utility = state.chance_prob() * sum([self._cfr_utility_recursive(outcome, reach_vector) for outcome in chance_outcomes])
                return shared_utility

        strategy = self.get_strategy(state)
        # sum up all utilities for playing actions in our game state
        for action in state.actions:
            reach_vector_child = np.copy(reach_vector)
            reach_vector_child[state.get_player_to_move().get_index()] *= strategy[action]

            action_utilities[action] = self._cfr_utility_recursive(state.play(action), reach_vector_child)

            for player in self._players:
                node_utilities[player.get_index()] += strategy[action] * action_utilities[action][player.get_index()]

        # accumulate regret
        for action in state.actions:
            # likelihood of arriving at this state given our strategy
            counterfactual = 1
            for player in self._players:
                if player != state.get_player_to_move():
                    counterfactual *= reach_vector[player.get_index()]

            player_index = state.get_player_to_move().get_index()
            regret = action_utilities[action][player_index] - node_utilities[player_index]

            info_set = state.inf_set()
            self.cumulative_regrets[info_set][action] += counterfactual * regret
            self.cumulative_sigma[info_set][action] += counterfactual * strategy[action]

        return node_utilities

    def __value_of_the_game_state_recursive(self, node):
        value = np.zeros(len(self._players))
        if node.is_terminal():
            return node.evaluation()
        for action in node.actions:
            if node.is_chance():
                player_index = 0
            else:
                player_index = node.get_player_to_move().get_index()
            value += self.nash_equilibrium[player_index][node.inf_set()][action] * self.__value_of_the_game_state_recursive(node.play(action))

        return value


class VanillaCFR(CounterfactualRegretMinimizationBase):

    def __init__(self, root, players):
        super().__init__(root=root, players=players, chance_sampling = False)

    def run(self, iterations=1):
        utilities = np.zeros(len(self._players))
        for _ in range(0, iterations):
            utilities += self._cfr_utility_recursive(self.root, np.ones(len(self._players)))

        # since we do not update sigmas in each information set while traversing, we need to
        # traverse the tree to perform to update it now
        #self.__update_sigma_recursively(self.root)

    def __update_sigma_recursively(self, node):
        # stop traversal at terminal node
        if node.is_terminal():
            return
        # omit chance
        if not node.is_chance():
            self.get_strategy(node)
        # go to subtrees
        for k in node.get_children():
            self.__update_sigma_recursively(node.get_children()[k])

class ChanceSamplingCFR(CounterfactualRegretMinimizationBase):

    def __init__(self, root, players):
        super().__init__(root=root, players=players, chance_sampling=True)

    def run(self, iterations=1):
        for _ in range(0, iterations):
            self._cfr_utility_recursive(self.root, np.ones(len(self._players)))

class ExternalSamplingCFR(CounterfactualRegretMinimizationBase):
    def __init__(self, root, players):
        super().__init__(root=root, players=players, chance_sampling=True)

    def run(self, iterations=1):
        utilities = np.zeros(len(self._players))
        for _ in range(0, iterations):
            sampling_memory = {}
            for player in self._players:
                utilities += self._perspective_cfr_utility_recursive(player, sampling_memory, self.root, np.ones(len(self._players)))
        return utilities

    def _perspective_cfr_utility_recursive(self, perspective, sampling_memory, state, reach_vector):
        action_utilities = {}
        node_utilities = np.zeros(len(self._players))
        info_set = state.inf_set()

        if state.is_terminal():
            return state.evaluation()
        if state.is_chance():
            return self._perspective_cfr_utility_recursive(perspective, sampling_memory, state.sample_one(), reach_vector)

        strategy = self.get_strategy(state)

        # if the player to move is not the player we're focused for perspective
        if state.get_player_to_move() != perspective:
            # sample a random action
            weight_vector = [strategy[action] for action in state.actions]
            #action_sampled = sampling_memory[info_set] if info_set in sampling_memory else np.random.choice(state.actions, p=weight_vector)
            action_sampled = np.random.choice(state.actions, p=weight_vector)

            # remember the action for future iterations on the same infoset
            #sampling_memory[info_set] = action_sampled

            # update reach vector for next iteration
            reach_vector_child = np.copy(reach_vector)
            reach_vector_child[state.get_player_to_move().get_index()] *= strategy[action_sampled]

            # no update to regrets
            return self._perspective_cfr_utility_recursive(perspective, sampling_memory,
                                                            state.play(action_sampled), reach_vector_child)

        # if player to move IS player we're focused on for perspective
        for action in state.actions:
            reach_vector_child = np.copy(reach_vector)
            reach_vector_child[state.get_player_to_move().get_index()] *= strategy[action]

            action_utilities[action] = self._perspective_cfr_utility_recursive(perspective, sampling_memory,
                                                                               state.play(action), reach_vector_child)

            node_utilities = strategy[action] * action_utilities[action]

        # accumulate regret
        for action in state.actions:
            # likelihood of arriving at this state given our strategy assuming our perspective player wanted to get there
            perspective_index = perspective.get_index()
            #print(reach_vector)
            counterfactual = 1
            for player in self._players:
                if player != state.get_player_to_move():
                    counterfactual *= reach_vector[player.get_index()]

            regret = action_utilities[action][perspective_index] - node_utilities[perspective_index]

            self.cumulative_regrets[info_set][action] += counterfactual * regret
            #self.cumulative_sigma[perspective_index][info_set][action] += counterfactual * strategy[action]

        return node_utilities

    def compute_nash_equilibrium(self):
        RuntimeError("Not Implemented for scability reasons")

    def run_simulation(self):
        curr_node = self.root
        while not curr_node.is_terminal():
            if curr_node.is_chance():
                curr_node = curr_node.play(random.choice(curr_node.actions))
            else:
                strategy = self.get_strategy(curr_node)
                weight_vector = [strategy[action] for action in curr_node.actions]
                curr_node = curr_node.play(np.random.choice(curr_node.actions, p=weight_vector))
        return curr_node.evaluation()

    def approximate_value_of_game(self, num_simulations=100):
        values = np.zeros(len(self._players))
        for i in range(num_simulations):
            values += self.run_simulation()
        return values / num_simulations