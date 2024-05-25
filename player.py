from abc import ABC, abstractmethod
from enum import Enum, unique
import pygame
from constants import CELL_SIZE

@unique
class PlayerID(Enum):
    """
    Enum defining the possible player IDs for the Othello game.
    """
    PLAYER_1 = 1
    PLAYER_2 = 2

class Player(ABC):
    """
    Abstract base class for a player in the Othello game.

    Attributes:
    id (int): The identifier for the player, typically 1 or 2.
    """
    
    def __init__(self, player_id):
        """
        Initializes the player with an ID.
        Restricts the player_id parameter to values defined in the PlayerID enum.
        """
        if not isinstance(player_id, PlayerID):
            raise ValueError("player_id must be a value from the PlayerID enum.")
        
        self.id = player_id
        
    def set_id(self, id):
        """
        Sets the player's ID.

        Parameters:
        id (int): The ID to be set for the player.
        """
        if not isinstance(id, PlayerID):
            raise ValueError("ID must be a value from the PlayerID enum.")
        
        self.id = id

    @abstractmethod
    def get_move(self, game, valid_moves, events):
        """
        Abstract method to be implemented by subclasses to get the player's move.

        Parameters:
        game (np.ndarray): The current state of the game board.
        valid_moves (list of tuple): A list of valid moves (row, col) for the player.
        events (list of pygame.event.Event): A list of Pygame events.

        Returns:
        tuple: The chosen move as a (row, col) tuple, or None if no valid move is chosen.
        """
        pass

class HumanPlayer(Player):
    """
    Class for a human player in the Othello game.
    """
    def get_move(self, game, valid_moves, events):
        """
        Gets the move from the human player based on mouse input.

        Parameters:
        game (np.ndarray): The current state of the game board.
        valid_moves (list of tuple): A list of valid moves (row, col) for the player.
        events (list of pygame.event.Event): A list of Pygame events.

        Returns:
        tuple: The chosen move as a (row, col) tuple, or None if no valid move is chosen.
        """
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                if (row, col) in valid_moves:
                    return row, col
        return None