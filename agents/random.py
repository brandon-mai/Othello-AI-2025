import random
from agents.player import Agent
from utils.array_utils import get_possible_moves


class RandomAgent(Agent):
    """
    Class for a Random Agent in the Othello game.
    """
    def __init__(self, id):
        super().__init__(id)

    def get_move(self, board, events):
        """
        Gets a random move from the epossible moves.

        Parameters:
        board (np.ndarray): The current state of the game board.
        events (list of pygame.event.Event): A list of Pygame events.

        Returns:
        tuple: The chosen move as a (row, col) tuple, or None if no valid move is chosen.
        """
        valid_moves = get_possible_moves(self.id, board)
        
        if valid_moves.shape[0] > 0:
            row, col = random.choice(valid_moves)
            return  row, col

        return None
    
    def copy(self):
        """
        Returns a new instance of RandomAgent.

        Returns:
        RandomAgent: A new instance with the same parameters.
        """
        return RandomAgent()