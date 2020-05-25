from game.kuhn import KuhnGame
from game.player import create_player_set
import pydealer
from tqdm import tqdm
from game.poker import PokerActions
from ld.liarsdice import LDGame

from optimizer.cfr import ChanceSamplingCFR, VanillaCFR, ExternalSamplingCFR


def run_game():
    deck = pydealer.Deck()

    cards = deck.get_list(['Jack of Spades', 'Queen of Spades', 'King of Spades'])
    players = create_player_set(2)

    game = KuhnGame(players, cards, 1)

    root = game.create_root_node()

    #players = create_player_set(2)
    #ldgame = LDGame(players, 1)
    #root = ldgame.create_root_node()

    cfr = ExternalSamplingCFR(root, players)
    for i in tqdm(range(1000)):
        cfr.run(iterations=1)
        #if i % 100 == 0:
        #    print(vanilla_cfr.compute_nash_equilibrium())
    print("Approx Value", cfr.approximate_value_of_game(1000))
    print(cfr._learned_strategy)

if __name__ == "__main__":
    run_game()