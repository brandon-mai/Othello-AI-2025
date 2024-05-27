from abc import ABC, abstractmethod

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