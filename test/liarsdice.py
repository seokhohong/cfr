import unittest

from game.liarsdice import LDGame, _get_ld_actions


class TestLDMethods(unittest.TestCase):
    def test_combinations(self):
        ldgame = LDGame(2, 5)
        self.assertEqual(len(list(ldgame.enumerate_possible_rolls())), 6 ** 5)

    def test_ld_actions(self):
        actions = _get_ld_actions((3, 3), 5)
        self.assertEqual(len(actions), 2 + 3 + 6 + 6)

if __name__ == '__main__':
    unittest.main()