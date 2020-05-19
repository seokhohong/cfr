import unittest

import numpy as np

from game import player, poker
from game.kuhn import KuhnGame, KuhnPlayerMoveGameState, ChanceGameState
from game.player import create_player_set

import pydealer
import random
import itertools

from game.poker import PokerActions
from optimizer.cfr import VanillaCFR, ChanceSamplingCFR


class TestKuhnMethods(unittest.TestCase):
    def test_game_tree(self):
        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])

        num_players = 2
        game = KuhnGame(create_player_set(num_players), cards, 1)

        # see if simulations run successfully
        for i in range(100):
            curr_node = game.create_root_node()
            while not curr_node.is_terminal():
                curr_node = random.choice(list(curr_node.get_children().values()))
            self.assertTrue(len(np.nonzero(curr_node.evaluation())[0]) == num_players)

    def test_possible_hands(self):
        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        game = KuhnGame(create_player_set(2), cards, 1)
        self.assertEqual(len(list(game.enumerate_possible_hands())), 6)

    def get_chance_node(self):
        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(2)

        game = KuhnGame(players, cards, 1)

        return game.create_root_node()

    def __recursive_tree_assert(self, root, logical_expression):
        assert logical_expression(root)
        for k in root.get_children():
            self.__recursive_tree_assert(root.get_children()[k], logical_expression)

    def test_kuhn_tree_actions_number_equal_to_children(self):
        root = self.get_chance_node()
        self.__recursive_tree_assert(root, lambda node: len(node.get_children()) == len(node.actions))

    def test_kuhn_to_move_chance_at_root(self):
        root = self.get_chance_node()
        assert root.get_player_to_move() == poker.ChancePlayer

    def test_kuhn_to_move_changes_correctly_for_children(self):
        logical_expression = lambda node: all([node.get_player_to_move() == node.get_children()[k].get_player_to_move().get_next() for k in node.get_children()])
        root = self.get_chance_node()
        for k in root.get_children():
            child = root.get_children()[k]
            self.__recursive_tree_assert(child, logical_expression)

    def test_player_a_acts_first(self):

        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(2)

        game = KuhnGame(players, cards, 1)
        root = game.create_root_node()

        for k in root.get_children():
            child = root.get_children()[k]
            assert child.get_player_to_move() == players[0]

    def test_if_only_root_is_chance(self):
        logical_expression = lambda node: not node.is_chance()
        root = self.get_chance_node()
        assert root.is_chance()
        for k in root.get_children():
            child = root.get_children()[k]
            self.__recursive_tree_assert(child, logical_expression)

    def test_if_possible_to_play_unavailable_action(self):
        root = self.get_chance_node()
        with self.assertRaises(KeyError):
            root.play(PokerActions.CALL)
        with self.assertRaises(KeyError):
            root.play(PokerActions.RAISE_1).play(PokerActions.RAISE_1)
        with self.assertRaises(KeyError):
            root.play(PokerActions.CHECK).play(PokerActions.CALL)


    def test_inf_sets(self):

        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        j, q, k = cards
        players = create_player_set(2)

        game = KuhnGame(players, cards, 1)

        root = game.create_root_node()

        KQ_node = root.get_children()[(k, q)]
        QJ_node = root.get_children()[(q, j)]
        KJ_node = root.get_children()[(k, j)]
        QK_node = root.get_children()[(q, k)]
        JQ_node = root.get_children()[(j, q)]
        JK_node = root.get_children()[(j, k)]

        assert root.inf_set() == "."
        self.assertTrue('King of Spades' in KQ_node.inf_set())
        self.assertTrue('Queen of Spades' in KQ_node.play(PokerActions.RAISE_1).inf_set())

    '''
    def test_termination():
        root = KuhnRootChanceGameState(CARDS_DEALINGS)
        assert not root.is_terminal()
        assert not root.play(KQ).play(BET).is_terminal()
        assert not root.play(JQ).play(CHECK).play(BET).is_terminal()
        assert not root.play(QJ).play(CHECK).is_terminal()

        assert root.play(KQ).play(BET).play(FOLD).is_terminal()
        assert root.play(JQ).play(CHECK).play(CHECK).is_terminal()
        assert root.play(JK).play(BET).play(CALL).is_terminal()
        assert root.play(QJ).play(CHECK).play(BET).play(FOLD).is_terminal()
        assert root.play(QJ).play(CHECK).play(BET).play(CALL).is_terminal()

    '''
    def test_evaluation(self):
        deck = pydealer.Deck()
        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        j, q, k = cards
        players = create_player_set(2)

        game = KuhnGame(players, cards, 1)

        root = game.create_root_node()

        KQ_node = root.get_children()[(k, q)]
        QJ_node = root.get_children()[(q, j)]
        KJ_node = root.get_children()[(k, j)]
        QK_node = root.get_children()[(q, k)]
        JQ_node = root.get_children()[(j, q)]
        JK_node = root.get_children()[(j, k)]

        for node in [KQ_node, QJ_node, KJ_node]:
            self.assertTrue(np.allclose(node.play(PokerActions.RAISE_1).play(PokerActions.FOLD).evaluation(), np.array([1, -1])))
            self.assertTrue(np.allclose(node.play(PokerActions.RAISE_1).play(PokerActions.CALL).evaluation(), np.array([2, -2])))
            self.assertTrue(np.allclose(node.play(PokerActions.CHECK).play(PokerActions.RAISE_1).play(PokerActions.FOLD).evaluation(), np.array([-1, 1])))
            self.assertTrue(np.allclose(node.play(PokerActions.CHECK).play(PokerActions.CHECK).evaluation(), np.array([1, -1])))

        for node in [QK_node, JQ_node, JK_node]:
            self.assertTrue(np.allclose(node.play(PokerActions.RAISE_1).play(PokerActions.FOLD).evaluation(), np.array([1, -1])))
            self.assertTrue(np.allclose(node.play(PokerActions.RAISE_1).play(PokerActions.CALL).evaluation(), np.array([-2, 2])))
            self.assertTrue(np.allclose(node.play(PokerActions.CHECK).play(PokerActions.RAISE_1).play(PokerActions.FOLD).evaluation(), np.array([-1, 1])))
            self.assertTrue(np.allclose(node.play(PokerActions.CHECK).play(PokerActions.CHECK).evaluation(), np.array([-1, 1])))

    def fresh_tkq_node(self):
        deck = pydealer.Deck()

        cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(3)

        game = KuhnGame(players, cards, num_deal=1)

        root = game.create_root_node()

        vanilla_cfr = VanillaCFR(root, players)
        t, j, q, k = cards

        TKQ_node = root.get_children()[(t, k, q)]
        return vanilla_cfr, TKQ_node

    def test_tkq_1(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(
            TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.CALL), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-2, 3.5, -1.5])))

    def test_tkq_2(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(
            TKQ_node.play(PokerActions.RAISE_1), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1, 1.25, -0.25])))

    def test_tkq_3(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node, np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1.1875, 1.78125, -0.59375])))

    def test_tkq_4(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.CHECK), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1.375, 2.3125, -0.9375]), atol=1E-1))

    def test_tkq_5(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.CHECK).play(PokerActions.CHECK).play(PokerActions.RAISE_1), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1.5, 1.25, 0.25]), atol=1E-1))

    def test_tkq_6(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.CHECK).play(PokerActions.CHECK), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1.25, 1.62, -0.38]), atol=1E-1))

    def test_tkq_7(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.CHECK).play(PokerActions.CHECK).play(PokerActions.CHECK), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-1, 2, -1])))

    def test_tkq_8(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), np.array([1, 1, 1]))
        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), np.array([1, 1, 1]))
        self.assertTrue(np.allclose(eval, np.array([-2, -1, 3])))

    def test_tkq_9(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.CALL), np.ones(3))
        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.CALL), np.ones(3))
        self.assertTrue(np.allclose(eval, np.array([-2, 3, -1])))

    def test_tkq_10(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), np.ones(3))
        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), np.ones(3))
        self.assertTrue(np.allclose(eval, np.array([-2, -1, 3])))

    def test_tkq_11(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1), np.ones(3))
        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), np.ones(3))
        self.assertTrue(np.allclose(eval, np.array([-2, -1, 3])))

    def test_tkq_12(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()

        vanilla_cfr._cfr_utility_recursive(TKQ_node, np.ones(3))
        vanilla_cfr._cfr_utility_recursive(TKQ_node, np.ones(3))
        eval = vanilla_cfr._cfr_utility_recursive(TKQ_node, np.ones(3))
        self.assertTrue(np.allclose(eval, np.array([-1.19, 2.19, -1]), atol=1E-1))

    def test_tjk_12(self):
        deck = pydealer.Deck()

        cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(3)

        game = KuhnGame(players, cards, num_deal=1)

        root = game.create_root_node()

        vanilla_cfr = VanillaCFR(root, players)
        t, j, q, k = cards

        TJQ_node = root.get_children()[(t, j, q)]
        TJK_node = root.get_children()[(t, j, k)]

        starting_reach = np.ones(3)
        vanilla_cfr._cfr_utility_recursive(TJQ_node, starting_reach)
        val = vanilla_cfr._cfr_utility_recursive(TJK_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD), starting_reach)
        #val = vanilla_cfr._cfr_utility_recursive(TJK_node, np.ones(3))
        #self.assertTrue(np.allclose(val, np.array([-2, 0.5, 1.5]), atol=1E-1))

    def test_full_iteration(self):
        deck = pydealer.Deck()

        cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(3)

        game = KuhnGame(players, cards, num_deal=1)

        root = game.create_root_node()

        vanilla_cfr = VanillaCFR(root, players)

        val = np.zeros(3)
        for permutation in itertools.permutations(cards, 3):
            one_val = vanilla_cfr._cfr_utility_recursive(root.get_children()[permutation], np.ones(3))
            val += one_val

        self.assertTrue(np.allclose(val, np.array([-0.61, 0.26, 0.35]), atol=1E-1))

    def test_chance(self):
        deck = pydealer.Deck()

        cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(3)

        game = KuhnGame(players, cards, num_deal=1)

        root = game.create_root_node()

        chance_cfr = ChanceSamplingCFR(root, players)
        chance_cfr.run(iterations=100)
        chance_cfr.compute_nash_equilibrium()

        game_value = chance_cfr.value_of_the_game()
        self.assertTrue(game_value[0] > -3. / 48)
        self.assertTrue(game_value[0] < -1. / 48)
        self.assertTrue(4. / 48 > game_value[2] > 2. / 48)

    def test_tkq_terminals(self):
        vanilla_cfr, TKQ_node = self.fresh_tkq_node()
        self.assertTrue(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.CALL).play(PokerActions.CALL).is_terminal())

    def fresh_kq_node(self):
        deck = pydealer.Deck()

        cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
        players = create_player_set(2)

        game = KuhnGame(players, cards, 1)

        root = game.create_root_node()

        vanilla_cfr = VanillaCFR(root, players)
        j, q, k = cards

        KQ_node = root.get_children()[(k, q)]
        return vanilla_cfr, KQ_node

    def test_kq_1(self):
        vanilla_cfr, KQ_node = self.fresh_kq_node()
        self.assertTrue(
            np.allclose(KQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD).evaluation(), np.array([1, -1])))

        self.assertTrue(np.allclose(vanilla_cfr._cfr_utility_recursive(
            KQ_node.play(PokerActions.CHECK), np.array([1, 1])), np.array([0.75, -0.75])))

    def test_kq_2(self):
        vanilla_cfr, KQ_node = self.fresh_kq_node()
        self.assertTrue(np.allclose(vanilla_cfr._cfr_utility_recursive(
            KQ_node.play(PokerActions.CHECK).play(PokerActions.RAISE_1), np.array([0.5, 0.5])), np.array([0.5, -0.5])))

    def test_kq_3(self):
        vanilla_cfr, KQ_node = self.fresh_kq_node()

        eval = vanilla_cfr._cfr_utility_recursive(
            KQ_node.play(PokerActions.RAISE_1), np.array([0.5, 0.5]))
        self.assertTrue(np.allclose(eval, np.array([1.5, -1.5])))

    def test_three_players(self):
        deck = pydealer.Deck()
        cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
        t, j, q, k = cards
        players = create_player_set(3)

        game = KuhnGame(players, cards, 1)

        root = game.create_root_node()

        TKQ_node = root.get_children()[(t, k, q)]
        JKQ_node = root.get_children()[(j, k, q)]

        for node in [TKQ_node, JKQ_node]:
            self.assertTrue(np.allclose(node.play(PokerActions.RAISE_1).play(PokerActions.FOLD).play(PokerActions.FOLD).evaluation(), np.array([2, -1, -1])))
            self.assertTrue(np.allclose(node.play(PokerActions.CHECK).play(PokerActions.RAISE_1).play(PokerActions.FOLD).play(PokerActions.CALL).evaluation(), np.array([-2, 3, -1])))

if __name__ == '__main__':
    unittest.main()