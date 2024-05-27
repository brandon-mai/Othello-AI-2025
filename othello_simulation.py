# othello_simulation.py
import multiprocessing
import numpy as np
from tqdm import tqdm
from agents.player import Agent, Player
from agents.minmax import MinimaxAgent
from othello import Othello
from utils.array_utils import get_possible_moves
from utils.constants import PLAYER_1, PLAYER_2

class OthelloSimulation:
    """
    A class to handle the simulation of Othello games without GUI.
    """
    def __init__(self, player1: Player, player2: Player):
        self.game = Othello(player1, player2)

    def run_simulation(self, num_simulations: int):
        """
        Runs the simulation for the specified number of games without GUI.
        """
        if not isinstance(self.game.player1, Agent) or not isinstance(self.game.player2, Agent):
            raise ValueError("Cannot simulate games with non Agent instances.")
        
        nb_cores = max(0, multiprocessing.cpu_count() - 1)
        
        print("=============== Othello Simulation ===============")
        print(f"Starting simulation on {nb_cores} cores.\n")
        
        with multiprocessing.Pool(processes=nb_cores) as pool:
            game_results = list(tqdm(pool.imap_unordered(self.simulate_game, 
                                                         ((self.game.player1.copy(), self.game.player2.copy()) for _ in range(num_simulations))), total=num_simulations))
        
        counts = {1: 0, 2: 0, 0: 0}
        for result in game_results:
            counts[result] += 1
                
        print("\n===================== Results ====================")
        print(f"Player 1 | Wins: {counts[self.game.player1.id]:<3}, Draws: {counts[0]:<3}")
        print(f"Player 2 | Wins: {counts[self.game.player2.id]:<3}, Draws: {counts[0]:<3}")


    @staticmethod
    def simulate_game(players: tuple) -> int:
        """
        Simulates a single game without GUI and returns the winner.

        Parameters:
        tuple(Player): A Tuple with both players

        Returns:
        int: 1 if player 1 wins, 2 if player 2 wins, 0 if it's a draw.
        """
        player1, player2 = players
        game = Othello(player1, player2)

        while True:
            current_player_valid_moves = game.get_possible_moves()
            if current_player_valid_moves.size == 0:
                game.switch_player()
                opponent_valid_moves = game.get_possible_moves()
                if opponent_valid_moves.size == 0:
                    break
                continue
            
            move = game.current_player.get_move(game.board, None)
            if move in current_player_valid_moves:
                game.make_move(*move)

        winner = game.get_winner()
        return winner

if __name__ == "__main__":
    simulation = OthelloSimulation(player1=MinimaxAgent(id=PLAYER_1, depth=5, time_limit=2, verbose=False, heuristic='hybrid'),
                                   player2=MinimaxAgent(id=PLAYER_2, depth=5, time_limit=2, verbose=False, heuristic='stability'))
    
    simulation.run_simulation(100)
