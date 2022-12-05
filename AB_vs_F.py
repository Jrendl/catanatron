"""This script will execute games and collect data for 
the best AlphaBetaPlayer against the valueFunction player"""
from typing import List, Dict, Tuple
from tqdm import tqdm #python progress bar library
from catanatron_core.catanatron import Game, Color
from catanatron_experimental.machine_learning.players import minimax, mcts, value

weights = value.DEFAULT_WEIGHTS

#weights to vary: public_vps, production, enemy_production, 
#   buildable_nodes, logest_road, 
#   hand_resources, hand_devs, army_size

# These values from ``values``
# DEFAULT_WEIGHTS = {
#     # Where to place. Note winning is best at all costs
#     "public_vps": 3e14,
#     "production": 1e8,
#     "enemy_production": -1e8,
#     "num_tiles": 1,
#     # Towards where to expand and when
#     "reachable_production_0": 0,
#     "reachable_production_1": 1e4,
#     "buildable_nodes": 1e3,
#     "longest_road": 10,
#     # Hand, when to hold and when to use.
#     "hand_synergy": 1e2,
#     "hand_resources": 1,
#     "discard_penalty": -5,
#     "hand_devs": 10,
#     "army_size": 10.1,
# }

def play_n(n: int, weights: Dict):
    players = [ minimax.AlphaBetaPlayer(color=Color.RED, prunning=True, params=weights),
                value.ValueFunctionPlayer(color=Color.BLUE)
            ]  
    win_count_AB = 0
    win_count_F = 0
    for i in tqdm(range(n)):
        game = Game(players)
        winner_color = game.play()
        if winner_color == Color.RED:
            win_count_AB += 1
        elif winner_color == Color.BLUE:
            win_count_F += 1
        else: 
            # if no one wins before the turn cutoff
            # we could change this to maybe replay the game? 
            # that way we always get n definitive wins
            pass
    if win_count_AB > win_count_F:
        return 'AB', [win_count_AB, win_count_F]
    elif win_count_F > win_count_AB:
        return 'F', [win_count_AB, win_count_F]
    else:
        return 'TIE', [win_count_AB, win_count_F]


def main():
    winner, counts = play_n(1000, weights)
    print(f"w:{winner}: {counts[0]}-{counts[1]}")


if __name__ == '__main__':
    main()
