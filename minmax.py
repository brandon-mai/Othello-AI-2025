from player import Player
from utils import *


class MinimaxPlayer(Player):
    def __init__(self, depth = 3):
        super().__init__()
        self.depth = depth
        
    def get_move(self, board, valid_moves, events):
        a, best_row, best_col = minimax(board, self.depth, INT16_NEGINF, INT16_POSINF, True, self.id)
        return (best_row, best_col)

@njit(int16(int16[:, :], int16), cache = True)
def evaluate(board, player_id):
    return np.count_nonzero(board == player_id) - np.count_nonzero(board == (2 if player_id == 1 else 1))

# Disabled this optimisation for now it's only a 3x
# @njit(UniTuple(int16, 3)(int16[:, :], int16, int16, int16, boolean, int16))
def minimax(board, depth, alpha, beta, maximizing_player, player_id):

    opponent_id = 1 if player_id == 2 else 2
    
    # Precompute the lists of moves to save perf
    max_player_moves = get_possible_moves(player_id, board)
    min_player_moves = get_possible_moves(opponent_id, board)
    
    # Depth 0 or Game Over (both players cannot move)
    if depth == 0 or (max_player_moves.size == 0 and min_player_moves.size == 0):
        return evaluate(board, player_id)*(depth+1), -1, -1.
    
    # If the current player cannot move but the opponent can
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