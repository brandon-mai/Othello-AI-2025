import pygame
from agents.player import Player
from utils.array_utils import get_possible_moves
from utils.constants import CELL_SIZE


class HumanPlayer(Player):
    """
    Class for a human player in the Othello game.
    """
    def get_move(self, board, events):
        """
        Gets the move from the human player based on mouse input.

        Parameters:
        board (np.ndarray): The current state of the game board.
        events (list of pygame.event.Event): A list of Pygame events.

        Returns:
        tuple: The chosen move as a (row, col) tuple, or None if no valid move is chosen.
        """
        
        valid_moves = get_possible_moves(self.id, board)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                if (row, col) in valid_moves:
                    return row, col
        return None