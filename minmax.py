from collections import namedtuple
import random
import time
from player import Player
from array_utils import *

EXACT = 2
UPPERBOUND = 1
LOWERBOUND = 0

TTEntry = namedtuple('TTEntry', 'value depth flag best_move')

@njit(int16[:, :](int16[:, :]), cache = True)
def rotate_90(board):
    return np.rot90(board)

@njit(int16[:, :](int16[:, :]), cache = True)
def rotate_180(board):
    return np.rot90(board, 2)

@njit(int16[:, :](int16[:, :]), cache = True)
def rotate_270(board):
    return np.rot90(board, 3)

@njit(int16[:, :](int16[:, :]), cache = True)
def reflect_horizontal(board):
    return np.flipud(board)

@njit(int16[:, :](int16[:, :]), cache = True)
def reflect_vertical(board):
    return np.fliplr(board)

@njit(int16[:, :](int16[:, :]), cache = True)
def reflect_diagonal(board):
    return np.transpose(board)

@njit(int16[:, :](int16[:, :]), cache = True)
def reflect_anti_diagonal(board):
    return np.fliplr(np.flipud(np.transpose(board)))

@njit(uint64(int16[:, :], uint64[:, :, :]), cache = True)
def compute_single_zobrist_hash(board, zobrist_table):
    """
    Compute the Zobrist hash for the given board state.
    
    Parameters:
    board (int16[:, :]): The current game board.
    zobrist_table (uint64[:, :, :]): The Zobrist table for hashing.
    
    Returns:
    uint64: The Zobrist hash value of the board.
    """
    h = np.uint64(0)
    for x in range(8):
        for y in range(8):
            piece = board[x, y]
            if piece != 0:
                h ^= zobrist_table[x, y, piece]
    return h

@njit
def compute_zobrist_hash(board, zobrist_table):
    """
    Compute the Zobrist hash for the given board state considering isomorphic boards.
    
    These include:
    1. 90-degree rotation
    2. 180-degree rotation
    3. 270-degree rotation
    4. Horizontal reflection
    5. Vertical reflection
    6. Diagonal reflection (main diagonal)
    7. Anti-diagonal reflection (secondary diagonal)
    
    Parameters:
    board (int16[:, :]): The current game board.
    zobrist_table (uint64[:, :, :]): The Zobrist table for hashing.
    
    Returns:
    uint64: The Zobrist hash value of the board considering symmetrical transformations.
    """
    hash = compute_single_zobrist_hash(board, zobrist_table)
    hash = min(hash, compute_single_zobrist_hash(rotate_90(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(rotate_180(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(rotate_270(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(reflect_horizontal(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(reflect_vertical(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(reflect_diagonal(board), zobrist_table))
    hash = min(hash, compute_single_zobrist_hash(reflect_anti_diagonal(board), zobrist_table))

    return hash

def initialize_zobrist():
    """
    Initialize the Zobrist table for hashing board states.
    """
    zobrist_table = np.zeros((8, 8, 3), dtype=np.uint64)
    random.seed(42)  # Use a fixed seed for reproducibility
    for x in range(8):
        for y in range(8):
            for k in range(3):  # 0: empty, 1: player1, 2: player2
                zobrist_table[x, y, k] = random.getrandbits(64)
    return zobrist_table




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
        depth (int): The depth to which the Minimax algorithm will search. Default is 3.
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
              
        Improved the performances of the algo with a Transposition Table and Zobrist Hash

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
        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.flag == EXACT:
                return tt_entry.value, tt_entry.best_move
            elif tt_entry.flag == LOWERBOUND:
                alpha = max(alpha, tt_entry.value)
            elif tt_entry.flag == UPPERBOUND:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value, tt_entry.best_move
        
        # Precompute the list of possible moves for the current player
        player_moves = get_possible_moves(player_id, board)
        opponent_moves = get_possible_moves(opponent_id, board)
        
        # Base case: depth 0 or no moves left for both players (game over)
        if depth == 0 or (player_moves.size == 0 and opponent_moves.size == 0):
            return color * evaluate(board, self.id) * (depth + 1), (-1, -1)

        # If the current player cannot move but the opponent can, pass the turn to the opponent
        if player_moves.size == 0:
            return -self.negamax(board, depth, -beta, -alpha, -color)[0], (-1, -1)

        max_eval = -32767
        best_move = (-1, -1)
        for r, c in player_moves:
            temp_board = np.copy(board)
            flip_tiles((r, c), player_id, temp_board)
            eval_state, _ = self.negamax(temp_board, depth - 1, -beta, -alpha, -color)
            eval_state = -eval_state
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


    