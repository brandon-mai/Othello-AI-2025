from player import Player
from utils import *

EXACT = 2
UPPERBOUND = 1
LOWERBOUND = 0


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
        self.transposition_table = {}
        
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
        a, best_row, best_col = self.minimax(board, self.depth, INT16_NEGINF, INT16_POSINF, True)
        return (best_row, best_col)
    
    # Disabled this optimisation for now it's only a 3x
    # @njit(UniTuple(int16, 3)(int16[:, :], int16, int16, int16, boolean, int16))
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """
        Implements the Minimax algorithm with alpha-beta pruning to determine the best move.

        Parameters:
        board (int16[:, :]): The current game board.
        depth (int16): The current search depth.
        alpha (int16): The alpha value for alpha-beta pruning.
        beta (int16): The beta value for alpha-beta pruning.
        maximizing_player (boolean): True if the current player is the maximizing player, False otherwise.

        Returns:
        UniTuple(int16, 3): A tuple containing the evaluation score, the best row, and the best column for the move.
        """
        # Determine opponent's ID
        opponent_id = 1 if self.id == 2 else 2
        # Precompute the lists of moves for both players
        max_player_moves = get_possible_moves(self.id, board)
        min_player_moves = get_possible_moves(opponent_id, board)
        
        # Base case: depth 0 or no moves left for both players (game over)
        if depth == 0 or (max_player_moves.size == 0 and min_player_moves.size == 0):
            return evaluate(board, self.id)*(depth+1), -1, -1
        
        # If the maximizing player cannot move but the minimizing player can
        if maximizing_player and max_player_moves.size == 0:
            return self.minimax(board, depth - 1, alpha, beta, False)
        if not maximizing_player and min_player_moves.size == 0:
            return self.minimax(board, depth - 1, alpha, beta, True)

        if maximizing_player:
            max_eval = -32767
            best_row, best_col = -1, -1
            for r, c in max_player_moves:
                temp_board = np.copy(board)
                flip_tiles((r, c), self.id, temp_board)
                eval_state, _, _ = self.minimax(temp_board, depth - 1, alpha, beta, False)
                if eval_state > max_eval:
                    max_eval = eval_state
                    best_row, best_col = r, c
                alpha = max(alpha, eval_state)
                if beta <= alpha:
                    break
            return max_eval, best_row, best_col
        else:
            min_eval = 32767
            best_row, best_col = -1, -1
            for r, c in min_player_moves:
                temp_board = np.copy(board)
                flip_tiles((r, c), opponent_id, temp_board)
                eval_state, _, _ = self.minimax(temp_board, depth - 1, alpha, beta, True)
                if eval_state < min_eval:
                    min_eval = eval_state
                    best_row, best_col = r, c
                beta = min(beta, eval_state)
                if beta <= alpha:
                    break
                
            return min_eval, best_row, best_col
        
    # def negamax(self, board, depth, alpha, beta, color, player_id):
    #     # Determine opponent's ID
    #     opponent_id = 1 if player_id == 2 else 2
        
    #     moves = get_possible_moves(player_id, board)
    #     opponent_moves = get_possible_moves(opponent_id, board)
        
    #     # Base case: depth 0 or no moves left for both players (game over)
    #     if depth == 0 or (moves.size == 0 and opponent_moves.size == 0):
    #         return color * evaluate(board, player_id) * (depth+1), -1, -1
        
    #     if moves.size == 0:
    #         return self.negamax(board, depth - 1, alpha, beta, -color, player_id)

    #     best_eval = -32767
    #     best_row, best_col = -1, -1
        
    #     for r, c in moves:
    #         temp_board = np.copy(board)
    #         flip_tiles((r, c), player_id if color == 1 else opponent_id, temp_board)
    #         eval_score, _, _ = self.negamax(temp_board, depth - 1, -beta, -alpha, -color, player_id)
    #         eval_score = -eval_score
    #         if eval_score > best_eval:
    #             best_eval = eval_score
    #             best_row, best_col = r, c
                
    #         alpha = max(alpha, eval_score)
    #         if alpha >= beta:
    #             break
            
    #     return best_eval, best_row, best_col


    