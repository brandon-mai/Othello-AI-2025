from abc import ABC, abstractmethod
import random

from pygame import MOUSEBUTTONDOWN

from utils.array_utils import get_possible_moves
from utils.constants import CELL_SIZE

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
        if player_id != 1 and player_id != 2:
            raise ValueError("Player ID must be either 1 or 2.")
        
        self.id = player_id
        
    def __repr__(self):
        """
        Returns a string representation of the Player object.

        Returns:
            str: A string representation including the player's class name and ID.
        """
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id})"
        
    def set_id(self, player_id):
        """
        Sets the player's ID.

        Parameters:
        player_id (int): The ID to be set for the player.
        """
        if player_id != 1 and player_id != 2:
            raise ValueError("Player ID must be either 1 or 2.")
        
        self.id = player_id

    @abstractmethod
    def get_move(self, board, events):
        """
        Abstract method to be implemented by subclasses to get the player's move.

        Parameters:
        board (np.ndarray): The current state of the game board.
        events (list of pygame.event.Event): A list of Pygame events.

        Returns:
        tuple: The chosen move as a (row, col) tuple, or None if no valid move is chosen.
        """
        raise NotImplementedError('You must implement get_move()')
   
class Agent(Player):
    """
    Class for a AI Agent in the Othello game.
    """
    @abstractmethod
    def copy(self):
        """
        Returns a new instance of the Player with the same parameters as the current instance.
        Used for simulations.

        Returns:
        Agent: A new instance with the same parameters.
        """
        raise NotImplementedError('You must implement copy()') 
    
    
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
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                if (row, col) in valid_moves:
                    return row, col
        return None
    
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