import numpy as np
from agents import Player
from numba import uint64

from bitboard_utils import make_move, possible_moves, get_moves_index, count_bits, get_player_board
from constants import PLAYER_1, PLAYER_2


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
        self.board = (uint64(0x0000000810000000), uint64(0x0000001008000000))
        
        # Create the players
        self.player1, self.player2 = player1, player2
        
        self.player1.set_id(PLAYER_1)
        self.player2.set_id(PLAYER_2)
        
        if self.player1.id == self.player2.id:
            raise ValueError('Players must have different IDs')
        
        self.current_player = self.player1
        self.current_player_moves = self.get_possible_moves()

    def switch_player(self):
        """
        Switches the current player to the other player.
        """
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
        self.current_player_moves = self.get_possible_moves()

    def make_move(self, move: int):
        """
        Makes a move by the current player and flips the appropriate tiles.

        Parameters:
        move (int): The index of the move.
        """
        self.board = make_move(self.board, move, self.current_player.id)
        self.switch_player()
        
    def get_winner(self):
        """
        Determines the winner of the game based on the piece counts.

        Returns:
        int:    ID of player 1 if player 1 wins, 
                ID of player 2 if player 2 wins,
                0 if it's a draw.
        """
        
        player1_board, player2_board = self.board
        
        count_player1 = count_bits(player1_board)
        count_player2 = count_bits(player2_board)
        if count_player1 > count_player2:
            return self.player1.id
        elif count_player1 < count_player2:
            return self.player2.id
        else:
            return 0

    def get_possible_moves(self):
        
        player_bb, opponent_bb = get_player_board(self.board, self.current_player.id)
        
        empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        player_moves = get_moves_index(possible_moves_bb_player)
        
        return player_moves

