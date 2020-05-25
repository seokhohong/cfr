import unittest

from liarsdice.liarsdice import LDGame, _get_ld_actions, LDAction, CALL, SPOT_ON
from game.player import create_player_set
import numpy as np
import random

class TestLDMethods(unittest.TestCase):

    def test_ld_actions(self):
        actions = _get_ld_actions((3, 3), 5)
        self.assertEqual(str(actions[0]), 'CALL')
        self.assertEqual(len(actions), 2 + 3 + 6 + 6)

    def test_ld_game(self):
        players = create_player_set(2)
        ldgame = LDGame(players, 2)
        root = ldgame.create_root_node()
        self.assertEqual(len(root.get_children()), 36)

    def test_dice_rolls(self):
        players = create_player_set(2)
        ldgame = LDGame(players, 2)
        root = ldgame.create_root_node()
        random_node = random.choice(list(root.get_children().values()))
        self.assertEqual(len(random_node.get_children()), 36)
        random_full_roll = random.choice(list(random_node.get_children().values()))
        self.assertEqual(len(random_full_roll.get_children()), 24)
        first_bet = random_full_roll.get_children()[LDAction(False, False, 1, 1)]
        self.assertEqual(len(first_bet.get_children()), 25)

    def test_game1(self):
        players = create_player_set(2)
        ldgame = LDGame(players, 2)
        root = ldgame.create_root_node()
        self.assertTrue(np.allclose(root.play((3, 4)).play((5, 1)).play_bet(3, 2).play(CALL).evaluation(), np.array([-0.5, 0.5])))
        self.assertTrue(
            np.allclose(root.play((3, 4)).play((5, 1)).play_bet(2, 3).play(CALL).evaluation(), np.array([0.5, -0.5])))
        self.assertTrue(
            np.allclose(root.play((3, 4)).play((5, 1)).play_bet(2, 3).play(SPOT_ON).evaluation(), np.array([-0.5, 0.5])))

    def test_game2(self):
        players = create_player_set(3)
        ldgame = LDGame(players, 3)
        root = ldgame.create_root_node()
        self.assertTrue(
            np.allclose(root.play((3, 4, 1)).play((5, 1, 1)).play((6, 6, 3)).play_bet(3, 2).play(CALL).evaluation(), np.array([1./3, -2./3, 1./3])))
        self.assertTrue(
            np.allclose(root.play((3, 4, 1)).play((5, 1, 1)).play((6, 6, 3)).play_bet(2, 1).play_bet(2, 6).play(CALL).evaluation(), np.array([1./3, 1./3, -2./3])))
        self.assertTrue(
            np.allclose(root.play((3, 4, 1)).play((5, 1, 1)).play((6, 6, 3)).play_bet(2, 1).play_bet(2, 6).play_bet(3, 6).play(CALL).evaluation(), np.array([1./3, 1./3, -2./3])))

        with self.assertRaises(KeyError):
            root.play((3, 4, 1)).play((5, 1, 1)).play((6, 6, 3)).play_bet(2, 1).play_bet(1, 1)

if __name__ == '__main__':
    unittest.main()