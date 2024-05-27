import numpy as np
from agents.player import Player
from utils.array_utils import flip_tiles, get_possible_moves


class Othello:
    """
    A class to handle the Othello game logic.

    Attributes:
    board (np.ndarray): The game board.
    player1 (Player): The first player.
    player2 (Player): The second player.
    current_player (Player): The player whose turn it is.
    """
    def __init__(self, player1: Player, player2: Player):
        """
        Initializes the Othello game with two players.

        Parameters:
        player1 (Player): The first player.
        player2 (Player): The second player.
        """
        self.board = np.zeros((8, 8), dtype=np.int16)
        self.board[3:5, 3:5] = [[2, 1], [1, 2]]  # Initial pieces
        
        # Create the players
        self.player1, self.player2 = player1, player2
        
        if self.player1.id == self.player2.id:
            raise ValueError('Players must have different IDs')
        
        self.current_player = self.player1

    def switch_player(self):
        """
        Switches the current player to the other player.
        """
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2

    def make_move(self, row: int, col: int):
        """
        Makes a move by the current player and flips the appropriate tiles.

        Parameters:
        row (int): The row index of the move.
        col (int): The column index of the move.
        """
        flip_tiles((row, col), self.current_player.id, self.board)
        self.switch_player()
        
    def get_winner(self):
        """
        Determines the winner of the game based on the piece counts.

        Returns:
        int:    ID of player 1 if player 1 wins, 
                ID of player 2 if player 2 wins,
                0 if it's a draw.
        """
        count_player1 = np.count_nonzero(self.board == self.player1.id)
        count_player2 = np.count_nonzero(self.board == self.player2.id)
        if count_player1 > count_player2:
            return self.player1.id
        elif count_player1 < count_player2:
            return self.player2.id
        else:
            return 0

    def get_possible_moves(self):
        return get_possible_moves(self.current_player.id, self.board)

