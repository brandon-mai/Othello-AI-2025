from player import Player
from utils import *


class MinimaxPlayer(Player):
    """
    A player that uses the Minimax algorithm with alpha-beta pruning to determine the best move.

    Attributes:
    depth (int): The depth to which the Minimax algorithm will search.
    """
    def __init__(self, depth = 3):
        """
        Initializes the MinimaxPlayer with a specified search depth.

        Parameters:
        depth (int): The depth to which the Minimax algorithm will search. Default is 3.
        """
        super().__init__()
        self.depth = depth
        
    def get_move(self, board, valid_moves, events):
        """
        Determines the best move for the player using the Minimax algorithm.

        Parameters:
        board (int16[:, :]): The current game board.
        valid_moves (list): A list of valid moves.
        events (list): A list of game events.

        Returns:
        tuple: The best move (row, col) for the player.
        """
        a, best_row, best_col = minimax(board, self.depth, INT16_NEGINF, INT16_POSINF, True, self.id)
        return (best_row, best_col)

@njit(int16(int16[:, :], int16), cache = True)
def evaluate(board, player_id):
    """
    Evaluates the board state for the given player.

    Parameters:
    board (int16[:, :]): The current game board.
    player_id (int16): The player's ID (1 or 2).

    Returns:
    int16: The evaluation score, calculated as the difference between the number of player's tiles and opponent's tiles.
    """
    return np.count_nonzero(board == player_id) - np.count_nonzero(board == (2 if player_id == 1 else 1))

# Disabled this optimisation for now it's only a 3x
# @njit(UniTuple(int16, 3)(int16[:, :], int16, int16, int16, boolean, int16))
def minimax(board, depth, alpha, beta, maximizing_player, player_id):
    """
    Implements the Minimax algorithm with alpha-beta pruning to determine the best move.

    Parameters:
    board (int16[:, :]): The current game board.
    depth (int16): The current search depth.
    alpha (int16): The alpha value for alpha-beta pruning.
    beta (int16): The beta value for alpha-beta pruning.
    maximizing_player (boolean): True if the current player is the maximizing player, False otherwise.
    player_id (int16): The player's ID (1 or 2).

    Returns:
    UniTuple(int16, 3): A tuple containing the evaluation score, the best row, and the best column for the move.
    """

    # Determine opponent's ID
    opponent_id = 1 if player_id == 2 else 2
    
    # Precompute the lists of moves for both players
    max_player_moves = get_possible_moves(player_id, board)
    min_player_moves = get_possible_moves(opponent_id, board)
    
    # Base case: depth 0 or no moves left for both players (game over)
    if depth == 0 or (max_player_moves.size == 0 and min_player_moves.size == 0):
        return evaluate(board, player_id)*(depth+1), -1, -1.
    
    # If the maximizing player cannot move but the minimizing player can
    if maximizing_player and max_player_moves.size == 0:
        return minimax(board, depth - 1, alpha, beta, False, player_id)
    if not maximizing_player and min_player_moves.size == 0:
        return minimax(board, depth - 1, alpha, beta, True, player_id)

    if maximizing_player:
        max_eval = -127
        best_row, best_col = -1, -1
        for r, c in max_player_moves:
            temp_board = np.copy(board)
            flip_tiles((r, c), player_id, temp_board)
            eval_state, _, _ = minimax(temp_board, depth - 1, alpha, beta, False, player_id)
            if eval_state > max_eval:
                max_eval = eval_state
                best_row, best_col = r, c
            alpha = max(alpha, eval_state)
            if beta <= alpha:
                break
        return max_eval, best_row, best_col
    else:
        min_eval = 127
        best_row, best_col = -1, -1
        for r, c in min_player_moves:
            temp_board = np.copy(board)
            flip_tiles((r, c), opponent_id, temp_board)
            eval_state, _, _ = minimax(temp_board, depth - 1, alpha, beta, True, player_id)
            if eval_state < min_eval:
                min_eval = eval_state
                best_row, best_col = r, c
            beta = min(beta, eval_state)
            if beta <= alpha:
                break
            
        return min_eval, best_row, best_col