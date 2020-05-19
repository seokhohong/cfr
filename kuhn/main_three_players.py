from game.kuhn import KuhnGame
from game.player import create_player_set
from game.poker import PokerActions
import pydealer
from tqdm import tqdm
import numpy as np

from optimizer.cfr import ChanceSamplingCFR, VanillaCFR

import random

def run_game():

    deck = pydealer.Deck()
    hand = pydealer.Stack()

    cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
    players = create_player_set(3)

    game = KuhnGame(players, cards, 1)

    root = game.create_root_node()

    #random.seed(1)

    #chance_sampling_cfr = ChanceSamplingCFR(root, players)
    #chance_sampling_cfr.run(iterations=1000)
    #chance_sampling_cfr.compute_nash_equilibrium()
    # read Nash-Equilibrum via chance_sampling_cfr.nash_equilibrium member
    # try chance_sampling_cfr.value_of_the_game() function to get value of the game (-1/18)
    #print(chance_sampling_cfr.value_of_the_game())

    # vanilla cfr
    vanilla_cfr = VanillaCFR(root, players)
    for i in tqdm(range(10)):
        vanilla_cfr.run(iterations=100)
        vanilla_cfr.compute_nash_equilibrium()
        #print(vanilla_cfr.value_of_the_game())

def fresh_tkq_node():
    deck = pydealer.Deck()

    cards = deck.get_list(['10 of Spades', 'Jack of Spades', 'Queen of Spades', 'King of Spades'])
    players = create_player_set(3)

    game = KuhnGame(players, cards, num_deal=1)

    root = game.create_root_node()

    vanilla_cfr = VanillaCFR(root, players)
    t, j, q, k = cards

    TKQ_node = root.get_children()[(t, k, q)]
    return vanilla_cfr, TKQ_node

def run_test():
    vanilla_cfr, TKQ_node = fresh_tkq_node()

    vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1), np.ones(3))
    eval = vanilla_cfr._cfr_utility_recursive(TKQ_node.play(PokerActions.RAISE_1).play(PokerActions.FOLD),
                                              np.ones(3))

if __name__ == "__main__":
    run_game()