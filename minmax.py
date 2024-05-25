from collections import namedtuple
import time
from zobrist_hashing import initialize_zobrist, compute_zobrist_hash

from player import Player
from array_utils import *
from heuristics import *
from func_timeout import func_timeout, FunctionTimedOut

EXACT = 2
UPPERBOUND = 1
LOWERBOUND = 0

TTEntry = namedtuple('TTEntry', 'value depth flag best_move')


@njit(int16[:,:](int16[:,:], int16, int16[:,:], UniTuple(int16, 2)), cache = True)
def sort_moves(board, player_id, moves, previous_best_move):
    """
    Sorts the moves based on their scores evaluated by the static weights heuristic.

    Parameters:
    - board (int16[:, :]): The current state of the board.
    - player_id (int16): The player's ID (1 or 2).
    - moves (int16[:,:]): Numpy array of possible moves.

    Returns:
    - int16[:,:] : Sorted array of moves.
    """
    move_scores = np.zeros(moves.shape[0], dtype=np.int16)
    
    for i in range(moves.shape[0]):
        r, c = moves[i]
        
        if (r, c) == previous_best_move:
            move_scores[i] = 9999
            continue
        
        temp_board = board.copy()
        flip_tiles((r, c), player_id, temp_board)
        score = static_weights_heuristic(temp_board, player_id)
        move_scores[i] = score
    
    sorted_indices = np.argsort(-move_scores)
    sorted_moves = moves[sorted_indices]
    
    return sorted_moves


class MinimaxPlayer(Player):
    """
    A player that uses the Minimax algorithm with alpha-beta pruning to determine the best move.

    Attributes:
    depth (int): The depth to which the Minimax algorithm will search.
    """
    def __init__(self, depth, time_limit = None):
        """
        Initializes the MinimaxPlayer with a specified search depth.

        Parameters:
        depth (int): The depth to which the Minimax algorithm will search.
        time_limit (float or int): If defined, Iterative Deepening will be applied with this as time constraint
        """
        super().__init__()
        self.depth = depth
        self.zobrist_table = initialize_zobrist()
        self.transposition_table = {}
        self.time_limit = time_limit
        
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
        if self.time_limit:
            return self.negamax_iterative_deepening_timed(board, self.depth, self.time_limit)[1]
        else:
            return self.negamax(board, self.depth, INT16_NEGINF, INT16_POSINF, 1)[1]

    
    
    def negamax_iterative_deepening_timed(self, board, max_depth, time_limit):
        """
        Add the Iterative Deepening Process to the Negamax algorithm with time constraint.

        Parameters:
            board (int16[:, :]): The current game board.
            max_depth (int16): The maximum search depth to iterate until.
            time_limit (float): The time limit in seconds for the search.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
                - int16: The evaluation score of the current board state.
                - tuple: The best move (row, column) determined by the algorithm.
        """
        start_time = time.perf_counter()
        best_move = None
        best_score = INT16_NEGINF
        
        for depth in range(1, max_depth + 1):
            best_score, best_move = self.negamax(board, depth, INT16_NEGINF, INT16_POSINF, 1)
            if time.perf_counter() - start_time >= time_limit:
                break
        return best_score, best_move
        
    def negamax(self, board, depth, alpha, beta, color):
        """
        Implements the Negamax algorithm with alpha-beta pruning to determine the best move.
        Negamax is a variant form of minimax that relies on the zero-sum property of a two-player game.
        It relies on the fact that : min(a, b) = -max(-b, -a) so Negamax uses a single perspective with score inversion.
              
        Improved the performances of the algo with a Transposition Table and Zobrist Hash.

        Parameters:
            board (int16[:, :]): The current game board.
            depth (int16): The current search depth.
            alpha (int16): The alpha value for alpha-beta pruning.
            beta (int16): The beta value for alpha-beta pruning.
            color (int): 1 if the current player is the maximizing player, -1 if the current player is the minimizing player.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
                    - int16: The evaluation score of the current board state.
                    - tuple: The best move (row, column) determined by the algorithm.
        """
        # Determine the player ID based on the color
        player_id = self.id if color == 1 else (1 if self.id == 2 else 2)
        opponent_id = 3 - player_id 
        
        # Compute the Zobrist hash of the current board
        zobrist_hash = compute_zobrist_hash(board, self.zobrist_table)
        tt_entry = self.transposition_table.get(zobrist_hash)
        
        alpha_orig = alpha
        
        # Check if the current state is in the transposition table
        if tt_entry is not None:
            if tt_entry.depth >= depth:
                if tt_entry.flag == EXACT:
                    return tt_entry.value, tt_entry.best_move
                elif tt_entry.flag == LOWERBOUND:
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.flag == UPPERBOUND:
                    beta = min(beta, tt_entry.value)
                if alpha >= beta:
                    return tt_entry.value, tt_entry.best_move
            # Try the best move stored in the transposition table entry first
            previous_best_move = tt_entry.best_move
        else:
            previous_best_move = (-1, -1)
            
        # Precompute the list of possible moves for the current player
        player_moves = get_possible_moves(player_id, board)
        opponent_moves = get_possible_moves(opponent_id, board)
        
        # Base case: depth 0 or no moves left for both players (game over)
        if depth == 0 or (player_moves.size == 0 and opponent_moves.size == 0):
            return color * evaluate_disc_parity(board, self.id) * (depth + 1), (-1, -1)

        # If the current player cannot move but the opponent can, pass the turn to the opponent
        if player_moves.size == 0:
            return -self.negamax(board, depth, -beta, -alpha, -color)[0], (-1, -1)
        
        # Moves are first randomized so we can get different games based on how ties are ordered
        np.random.shuffle(player_moves)
        sorted_moves = sort_moves(board, player_id, player_moves, previous_best_move)

        max_eval = -32767
        best_move = (-1, -1)
        for r, c in sorted_moves:
            temp_board = np.copy(board)
            flip_tiles((r, c), player_id, temp_board)
            eval_state = -self.negamax(temp_board, depth - 1, -beta, -alpha, -color)[0]
            if eval_state > max_eval:
                max_eval = eval_state
                best_move = (r, c)
            alpha = max(alpha, eval_state)
            if alpha >= beta:
                break
        
            
        flag = EXACT
        if max_eval <= alpha_orig:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
            
        # Store the result in the transposition table
        self.transposition_table[zobrist_hash] = TTEntry(max_eval, depth, flag, best_move)
        
        return max_eval, best_move


    