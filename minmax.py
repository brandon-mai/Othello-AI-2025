import collections
import random
import numpy as np
from numba import njit, int32, int8, uint64, types, int16, int64
from numba.core.types import NamedTuple
from numba.types import UniTuple, DictType
from numba.typed import Dict
from numba.experimental import structref
from numba.extending import overload
from constants import INT16_NEGINF, INT16_POSINF
from heuristics import static_weights_heuristic, hybrid_heuristic

from bitboard_utils import get_moves_index, possible_moves, make_move, get_player_board

EXACT = int8(2)
UPPERBOUND = int8(1)
LOWERBOUND = int8(0)
    

# ================ Zobrist Hashing ================

@njit(int64[:, :](), cache = True)
def initialize_zobrist():
    """
    Initialize the Zobrist table for hashing board states.
    
    Returns:
        np.ndarray: The Zobrist table initialized with random values.
    """
    zobrist_table = np.zeros((64, 2), dtype=np.int64)
    random.seed(42)  # Use a fixed seed for reproducibility
    for pos in range(64):
        for state in range(2):  # 1: player1, 2: player2
            zobrist_table[pos, state] = random.getrandbits(64)
    return zobrist_table

@njit(int64(UniTuple(uint64, 2), int64[:, :]), cache=True)
def compute_zobrist_hash(boards, zobrist_table):
    """
    Compute the Zobrist hash for the given board state.

    Parameters:
        board (UniTuple(uint64, 2)): The current game board.
        zobrist_table (int64[:, :]): The Zobrist table for hashing.

    Returns:
        int64: The Zobrist hash value of the board.
    """
    hash_value = int64(0)
    bitboard_player1, bitboard_player2 = boards
    
    for pos in range(64):
        mask = int64(1) << pos
        if bitboard_player1 & mask:
            hash_value ^= zobrist_table[pos, 0]  # Player 1
        elif bitboard_player2 & mask:
            hash_value ^= zobrist_table[pos, 1]  # Player 2
    
    return hash_value

# ================ Transposition Table ================

# Define the TTEntry namedtuple using NamedTuple from numba.core.types
TTEntry = collections.namedtuple('TTEntry', 'value depth flag best_move')
TTEntryType = NamedTuple((int16, int8, int8, int8), TTEntry)

# Initialize a Numba dictionary with uint64 keys and TTEntry values
@njit(cache = True)
def initialize_tt_dict():
    """
    Initialize the transposition table dictionary.

    Returns:
        DictType(int64, TTEntryType): Empty dictionary for transposition table.
    """
    tt_dict = Dict.empty(
        key_type=types.int64,
        value_type=TTEntryType
    )
    return tt_dict
    
# =============== Move Sorting ===============
    
@njit(int8[::1](UniTuple(uint64, 2), int8[::1], int8, int8), cache = True)
def sort_moves(board, moves, previous_best_move, player_id):
    """
    Sorts the moves based on their scores evaluated by the static weights heuristic.

    Parameters:
    - board (UniTuple(uint64, 2)): The current state of the board for both players.
    - moves (int8[::1]): Numpy array of possible moves.
    - previous_best_move (int8): The previously computed best move for that position
    - player_id (int8): The ID of the current player

    Returns:
    - int8[::1] : Sorted array of moves.
    """
    move_scores = np.zeros(moves.shape[0], dtype=np.int16)
    
    for i in range(moves.shape[0]):
        m = moves[i]
        
        if m == previous_best_move:
            move_scores[i] = 9999
            continue
        
        new_board = make_move(board, m, player_id)
        score = static_weights_heuristic(new_board, player_id)
        move_scores[i] = score
    
    sorted_indices = np.argsort(-move_scores)
    sorted_moves = moves[sorted_indices]
    
    return sorted_moves
    
# ================ Minmax ================

class Minmax(structref.StructRefProxy):
    """
    Class implementing the Negamx algorithm with alpha-beta pruning for game AI.

    Attributes:
    - player_id (int8): ID of the player for whom the AI is making decisions.
    - zobrist_table (np.ndarray): Zobrist table for hashing board states.
    - transposition_table (DictType(int64, TTEntryType)): Transposition table for storing evaluated game states.
    """
    
    def __new__(cls, player_id):
        self = minmax_ctor(player_id)
        return self
    
    @property
    def zobrist_table(self):
        return _zobrist_table(self)
    
    @property
    def player_id(self):
        return _player_id(self)
    
    @property
    def transposition_table(self):
        return _transposition_table(self)
    
    def negamax(self, board, depth, alpha, beta, color):
        return _negamax(self, board, depth, alpha, beta, color)

@njit(cache = True)
def _player_id(self):
    return self.player_id
    
@njit(cache = True)
def _zobrist_table(self):
    return self.zobrist_table

@njit(cache = True)
def _transposition_table(self):
    return self.transposition_table

# ================ Python - Numba interfacing ================

minmax_fields = [
    ('player_id', int8),
    ('zobrist_table', int64[:, :]),
    ('transposition_table', DictType(int64, TTEntryType))
]

@structref.register
class MinmaxTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
structref.define_boxing(MinmaxTypeTemplate, Minmax)    
MinmaxType = MinmaxTypeTemplate(minmax_fields)

# ============================================================

@njit(UniTuple(int16, 2)(MinmaxType, UniTuple(uint64, 2), int8, int16, int16, int8))
def _negamax(self, board, depth, alpha, beta, color):
        """
        Implements the Negamax algorithm with alpha-beta pruning to determine the best move.
        Negamax is a variant form of minimax that relies on the zero-sum property of a two-player game.
        It relies on the fact that : min(a, b) = -max(-b, -a) so Negamax uses a single perspective with score inversion.
            
        Improved the performances of the algo with a Transposition Table and Zobrist Hash. Also added move ordering
        based on the static weight heuristic score.

        Parameters:

            self (MinmaxType): The Minmax structure that contains all the data
            board (UniTuple(uint64,)): The current game state.
            depth (int8): The current search depth.
            alpha (int16): The alpha value for alpha-beta pruning.
            beta (int16): The beta value for alpha-beta pruning.
            color (int8): 1 if the current player is the maximizing player, -1 if the current player is the minimizing player.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
                    - int16: The evaluation score of the current board state.
                    - int16: The best move determined by the algorithm.
        """
        
        current_player_id = self.player_id if color == 1 else 3 - self.player_id
        
        player_bb, opponent_bb = get_player_board(board, current_player_id)
        
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
            previous_best_move = int8(tt_entry.best_move)
        else:
            previous_best_move = -1
            
        # Precompute the list of possible moves for the current player
        empty_squares = np.uint64(player_bb | opponent_bb) ^ np.uint64(0xFFFFFFFFFFFFFFFF)
        
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        player_moves = get_moves_index(possible_moves_bb_player)
        
        possible_moves_bb_opponent = possible_moves(opponent_bb, player_bb, empty_squares)
        opponent_moves = get_moves_index(possible_moves_bb_opponent)

        # Base case: depth 0 or no moves left for both players (game over)
        if depth == 0 or (player_moves.shape[0] == 0 and opponent_moves.shape[0] == 0):
            return color * hybrid_heuristic(board, self.player_id), -1

        # If the current player cannot move but the opponent can, pass the turn to the opponent
        if player_moves.shape[0] == 0:
            return -_negamax(self, board, depth, -beta, -alpha, -color)[0], -1
        
        # Moves are first randomized so we can get different games based on how ties are ordered
        np.random.shuffle(player_moves)
        sorted_moves = sort_moves(board, player_moves, previous_best_move, current_player_id)

        max_eval = INT16_NEGINF
        best_move = int8(-1)
        for m in sorted_moves:
            new_board = make_move(board, m, current_player_id)
            eval_state = -_negamax(self, new_board, depth - 1, -beta, -alpha, -color)[0]
            if eval_state > max_eval:
                max_eval = eval_state
                best_move = m
            alpha = max(alpha, eval_state)
            if alpha >= beta:
                break
        
            
        flag = EXACT
        if max_eval <= alpha_orig:
            flag = UPPERBOUND
        elif max_eval >= beta:
            flag = LOWERBOUND
            
        # Store the result in the transposition table
        self.transposition_table[zobrist_hash] = TTEntry(int16(max_eval), int8(depth), flag, int8(best_move))
        
        return max_eval, best_move

@njit(MinmaxType(int8), cache=True)
def minmax_ctor(player_id):
    st = structref.new(MinmaxType)
    
    st.zobrist_table = initialize_zobrist()
    st.transposition_table = initialize_tt_dict()
    st.player_id = player_id
    
    return st

@overload(Minmax)
def overload_Minmax(player_id):
    def impl(player_id):
        return minmax_ctor(player_id)
    return impl