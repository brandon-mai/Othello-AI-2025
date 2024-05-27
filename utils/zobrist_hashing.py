from utils.constants import *
import random

@njit(int16[:, :](int16[:, :]), cache=True)
def rotate_90(board):
    """
    Rotate the given board 90 degrees clockwise.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board rotated 90 degrees clockwise.
    """
    return np.rot90(board)

@njit(int16[:, :](int16[:, :]), cache=True)
def rotate_180(board):
    """
    Rotate the given board 180 degrees.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board rotated 180 degrees.
    """
    return np.rot90(board, 2)

@njit(int16[:, :](int16[:, :]), cache=True)
def rotate_270(board):
    """
    Rotate the given board 270 degrees clockwise.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board rotated 270 degrees clockwise.
    """
    return np.rot90(board, 3)

@njit(int16[:, :](int16[:, :]), cache=True)
def reflect_horizontal(board):
    """
    Reflect the given board horizontally.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board reflected horizontally.
    """
    return np.flipud(board)

@njit(int16[:, :](int16[:, :]), cache=True)
def reflect_vertical(board):
    """
    Reflect the given board vertically.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board reflected vertically.
    """
    return np.fliplr(board)

@njit(int16[:, :](int16[:, :]), cache=True)
def reflect_diagonal(board):
    """
    Reflect the given board along the main diagonal.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board reflected along the main diagonal.
    """
    return np.transpose(board)

@njit(int16[:, :](int16[:, :]), cache=True)
def reflect_anti_diagonal(board):
    """
    Reflect the given board along the anti-diagonal.

    Parameters:
    board (int16[:, :]): The current game board.

    Returns:
    int16[:, :]: The board reflected along the anti-diagonal.
    """
    return np.fliplr(np.flipud(np.transpose(board)))

@njit(uint64(int16[:, :], uint64[:, :, :]), cache=True)
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

@njit(uint64(int16[:, :], uint64[:, :, :]), cache=True)
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

    # ========== Disabled ==========
    # hash = min(hash, compute_single_zobrist_hash(rotate_90(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(rotate_180(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(rotate_270(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(reflect_horizontal(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(reflect_vertical(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(reflect_diagonal(board), zobrist_table))
    # hash = min(hash, compute_single_zobrist_hash(reflect_anti_diagonal(board), zobrist_table))

    return hash

def initialize_zobrist():
    """
    Initialize the Zobrist table for hashing board states.

    Returns:
    np.ndarray: The Zobrist table initialized with random values.
    """
    zobrist_table = np.zeros((8, 8, 3), dtype=np.uint64)
    random.seed(42)  # Use a fixed seed for reproducibility
    for x in range(8):
        for y in range(8):
            for k in range(3):  # 0: empty, 1: player1, 2: player2
                zobrist_table[x, y, k] = random.getrandbits(64)
    return zobrist_table