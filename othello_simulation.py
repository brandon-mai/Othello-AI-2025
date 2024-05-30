# othello_simulation.py
import multiprocessing
import time
import numpy as np
from tqdm import tqdm
from agents import Agent, Player, MinimaxAgent, MCTSAgent, RandomAgent, HumanPlayer
from othello import Othello
from utils.constants import PLAYER_1, PLAYER_2
from numba import njit, int16

class OthelloSimulation:
    """
    A class to handle the simulation of Othello games without a graphical user interface (GUI).

    Attributes:
        game (Othello): An instance of the Othello game with the provided players.
    """
    
    def __init__(self, player1: Player, player2: Player):
        """
        Initializes the OthelloSimulation with two players.

        Args:
            player1 (Player): The first player.
            player2 (Player): The second player.
        """
        self.game = Othello(player1, player2)

    def run_simulation(self, num_simulations: int, parallel: bool = True):
        """
        Runs the simulation for the specified number of games without GUI.

        Args:
            num_simulations (int): The number of games to simulate.

        Raises:
            ValueError: If either of the players is not an instance of the Agent class.
        """
        if not isinstance(self.game.player1, Agent) or not isinstance(self.game.player2, Agent):
            raise ValueError("Cannot simulate games with non Agent instances.")
        
        nb_cores = max(0, multiprocessing.cpu_count() - 1)
        game_results = []
        
        print("=============== Othello Simulation ===============")
        start = time.perf_counter()
        if parallel:
            print(f"Starting simulation on {nb_cores} cores.\n")
            
            with multiprocessing.Pool(processes=nb_cores) as pool:
                game_results = list(tqdm(pool.imap_unordered(self.simulate_game, 
                                                            ((self.game.player1.copy(), self.game.player2.copy()) for _ in range(num_simulations))), total=num_simulations))
        else:
            for i in range(num_simulations):
                mid = time.perf_counter()
                result = self.simulate_game((self.game.player1.copy(), self.game.player2.copy())) 
                tot = time.perf_counter() - mid
                
                game_results.append(result)
                print(f"Simulation {i+1} took {tot:<6.2f} sec")
                
        
        end_tot = time.perf_counter() - start
        counts = {1: 0, 2: 0, 0: 0}
        for result in game_results:
            counts[result] += 1
                
        print("\n===================== Results ====================")
        print(f"Player 1 | Wins: {counts[self.game.player1.id]:<3}, Draws: {counts[0]:<3}")
        print(f"Player 2 | Wins: {counts[self.game.player2.id]:<3}, Draws: {counts[0]:<3}")
        print(f"Simulation took {end_tot:<7.2f} sec (avg:{end_tot/num_simulations:.2f})")

    @staticmethod
    def simulate_game(players: tuple) -> int:
        """
        Simulates a single game without GUI and returns the winner.

        Args:
            players (tuple(Player)): A tuple containing two player instances.

        Returns:
            int: The ID of the winner, or 0 if the game is a draw.
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
    simulation = OthelloSimulation(player1=MinimaxAgent(id=PLAYER_1, depth=9),
                                   player2=MCTSAgent(id=PLAYER_2, nb_iterations=10000))
    
    simulation.run_simulation(num_simulations=10, parallel=False)
    
