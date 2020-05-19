from game.kuhn import KuhnGame
from game.player import create_player_set
import pydealer
from tqdm import tqdm
from game.poker import PokerActions

from optimizer.cfr import ChanceSamplingCFR, VanillaCFR

import random

def run_game():

    deck = pydealer.Deck()

    cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
    players = create_player_set(2)

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

    root = game.create_root_node()
    j, q, k = cards

    KQ_node = root.get_children()[(k, q)]

    vanilla_cfr._cfr_utility_recursive(KQ_node.play(PokerActions.CHECK).play(PokerActions.CHECK), [1, 1])
    for i in tqdm(range(10)):
        vanilla_cfr.run(iterations=100)
        vanilla_cfr.compute_nash_equilibrium()
        print(vanilla_cfr.value_of_the_game())

if __name__ == "__main__":
    run_game()