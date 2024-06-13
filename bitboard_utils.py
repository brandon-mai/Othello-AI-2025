from numba import njit, void, uint64, int8, int16
from numba.types import UniTuple
import numpy as np

LEFT_MASK = 0x7F7F7F7F7F7F7F7F
RIGHT_MASK = 0xFEFEFEFEFEFEFEFE

@njit(int8(uint64), cache = True)
def count_bits(x):
    # https://www.chessprogramming.org/Population_Count
    k1 = 0x5555555555555555
    k2 = 0x3333333333333333
    k4 = 0x0f0f0f0f0f0f0f0f
    kf = 0x0101010101010101

    x =  x       - ((x >> 1)  & k1)
    x = (x & k2) + ((x >> 2)  & k2)
    x = (x       +  (x >> 4)) & k4 
    x = (x * kf) >> 56 
    return x

@njit(int8(uint64), cache = True)
def bit_scan_forward(bit_board):
    # https://www.chessprogramming.org/BitScan
    return count_bits((bit_board & -bit_board) - 1)

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_up(player_disks, opponent_disks, empty_squares):
    potential = (player_disks >> 8) & opponent_disks

    potential = (potential >> 8); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = (potential >> 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential >> 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential >> 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential >> 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential >> 8); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_up_right(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks >> 7) & RIGHT_MASK) & opponent_disks

    potential = ((potential >> 7) & RIGHT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 7) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 7) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 7) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 7) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 7) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_right(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks << 1) & RIGHT_MASK) & opponent_disks

    potential = ((potential << 1) & RIGHT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 1) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 1) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 1) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 1) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 1) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_down_right(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks << 9) & RIGHT_MASK) & opponent_disks

    potential = ((potential << 9) & RIGHT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 9) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 9) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 9) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 9) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 9) & RIGHT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_down(player_disks, opponent_disks, empty_squares):
    potential = (player_disks << 8) & opponent_disks

    potential = (potential << 8); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = (potential << 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential << 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential << 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential << 8); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = (potential << 8); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_down_left(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks << 7) & LEFT_MASK) & opponent_disks

    potential = ((potential << 7) & LEFT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 7) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 7) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 7) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 7) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential << 7) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_left(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks >> 1) & LEFT_MASK) & opponent_disks

    potential = ((potential >> 1) & LEFT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 1) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 1) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 1) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 1) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 1) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def validate_up_left(player_disks, opponent_disks, empty_squares):
    potential = ((player_disks >> 9) & LEFT_MASK) & opponent_disks

    potential = ((potential >> 9) & LEFT_MASK); valid_plays  = potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 9) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 9) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 9) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 9) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks
    potential = ((potential >> 9) & LEFT_MASK); valid_plays |= potential & empty_squares; potential &= opponent_disks

    return valid_plays

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def possible_moves(player_disks, opponent_disks, empty_squares):
    # https://www.chessprogramming.org/Dumb7Fill
    return (validate_up(player_disks, opponent_disks, empty_squares) |
            validate_up_right(player_disks, opponent_disks, empty_squares) |
            validate_right(player_disks, opponent_disks, empty_squares) |
            validate_down_right(player_disks, opponent_disks, empty_squares) |
            validate_down(player_disks, opponent_disks, empty_squares) |
            validate_down_left(player_disks, opponent_disks, empty_squares) |
            validate_left(player_disks, opponent_disks, empty_squares) |
            validate_up_left(player_disks, opponent_disks, empty_squares))
    
@njit(int8[::1](uint64), cache=True, nogil=True)
def get_moves_index(bitboard):
    indices = np.zeros(count_bits(bitboard), dtype=np.int8)
    count = 0
    while bitboard:
        index = bit_scan_forward(bitboard)
        indices[count] = index
        count += 1
        bitboard &= bitboard - uint64(1)
    return indices

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_up(placement, player_disks, opponent_disks):
    flipped_pieces = (placement >> 8) & opponent_disks

    flipped_pieces |= (flipped_pieces >> 8) & opponent_disks
    flipped_pieces |= (flipped_pieces >> 8) & opponent_disks
    flipped_pieces |= (flipped_pieces >> 8) & opponent_disks
    flipped_pieces |= (flipped_pieces >> 8) & opponent_disks
    flipped_pieces |= (flipped_pieces >> 8) & opponent_disks
    
    if ((flipped_pieces >> 8) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_up_right(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement >> 7) & RIGHT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces >> 7) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 7) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 7) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 7) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 7) & RIGHT_MASK) & opponent_disks
    
    if (((flipped_pieces >> 7) & RIGHT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_right(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement << 1) & RIGHT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces << 1) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 1) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 1) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 1) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 1) & RIGHT_MASK) & opponent_disks
    
    if (((flipped_pieces << 1) & RIGHT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_down_right(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement << 9) & RIGHT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces << 9) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 9) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 9) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 9) & RIGHT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 9) & RIGHT_MASK) & opponent_disks
    
    if (((flipped_pieces << 9) & RIGHT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_down(placement, player_disks, opponent_disks):
    flipped_pieces = (placement << 8) & opponent_disks

    flipped_pieces |= (flipped_pieces << 8) & opponent_disks
    flipped_pieces |= (flipped_pieces << 8) & opponent_disks
    flipped_pieces |= (flipped_pieces << 8) & opponent_disks
    flipped_pieces |= (flipped_pieces << 8) & opponent_disks
    flipped_pieces |= (flipped_pieces << 8) & opponent_disks
    
    if ((flipped_pieces << 8) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_down_left(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement << 7) & LEFT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces << 7) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 7) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 7) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 7) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces << 7) & LEFT_MASK) & opponent_disks
    
    if (((flipped_pieces << 7) & LEFT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_left(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement >> 1) & LEFT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces >> 1) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 1) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 1) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 1) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 1) & LEFT_MASK) & opponent_disks
    
    if (((flipped_pieces >> 1) & LEFT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_up_left(placement, player_disks, opponent_disks):
    flipped_pieces = ((placement >> 9) & LEFT_MASK) & opponent_disks

    flipped_pieces |= ((flipped_pieces >> 9) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 9) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 9) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 9) & LEFT_MASK) & opponent_disks
    flipped_pieces |= ((flipped_pieces >> 9) & LEFT_MASK) & opponent_disks
    
    if (((flipped_pieces >> 9) & LEFT_MASK) & player_disks) == 0:
        return 0

    return flipped_pieces

@njit(uint64(uint64, uint64, uint64), cache=True, nogil=True)
def place_disks(placement, player_disks, opponent_disks):
    return (place_up(placement, player_disks, opponent_disks) |
            place_up_right(placement, player_disks, opponent_disks) |
            place_right(placement, player_disks, opponent_disks) |
            place_down_right(placement, player_disks, opponent_disks) |
            place_down(placement, player_disks, opponent_disks) |
            place_down_left(placement, player_disks, opponent_disks) |
            place_left(placement, player_disks, opponent_disks) |
            place_up_left(placement, player_disks, opponent_disks))

@njit(uint64(UniTuple(uint64, 2), int8), cache=True)
def find_empty_neighbors_of_player(board, player_id):
    """
    Find the empty neighboring cells of a player on the game board.

    Parameters:
    - board (UniTuple): A tuple containing two uint64 bitboards representing the game state.
    - player_id (int16): The ID of the player whose empty neighboring cells are to be found.

    Returns:
    - uint64: A bitboard representing the empty neighboring cells of the specified player.
    """
    # Get the bitboards of the current and opponent players
    player_board = board[player_id - 1]
    opponent_board = board[2 - player_id]
    
    # Combine both players' bitboards
    all_players_board = player_board | opponent_board  
    
    # Compute the set of empty cells
    empty_board = all_players_board ^ 0xFFFFFFFFFFFFFFFF
    
    # Compute empty neighboring cells
    empty_neighbors = empty_board & (
        ((player_board >> 8)) |  # Shift north
        ((player_board << 8)) |  # Shift south
        ((player_board << 1) & RIGHT_MASK) |  # Shift east
        ((player_board >> 1) & LEFT_MASK) |  # Shift west
        ((player_board >> 7) & RIGHT_MASK) |  # Shift northeast
        ((player_board >> 9) & LEFT_MASK) |  # Shift northwest
        ((player_board << 9) & RIGHT_MASK) |  # Shift southeast
        ((player_board << 7) & LEFT_MASK)    # Shift southwest
    )
    
    return empty_neighbors

def visualize_bitboard(player1_pieces, player2_pieces=0):
    board = np.zeros((8, 8), dtype=str)
    board.fill('.')
    
    for i in range(64):
        if (uint64(player1_pieces) >> uint64(i)) & uint64(1):
            row, col = divmod(i, 8)
            board[row, col] = '1'
        elif (uint64(player2_pieces) >> uint64(i)) & uint64(1):
            row, col = divmod(i, 8)
            board[row, col] = '2'
    
    print("  A B C D E F G H")
    print(" +----------------")
    for i, row in enumerate(board):
        print(f"{i}|{' '.join(row)}|")
    print(" +----------------")
    print("  A B C D E F G H")
    
@njit(UniTuple(uint64, 2)(UniTuple(uint64, 2), int8), cache = True)
def get_player_board(board, player_id):
    bb1, bb2 = board
    if player_id == 1:
        return bb1, bb2
    return bb2, bb1

@njit(UniTuple(uint64, 2)(UniTuple(uint64, 2), int8, int8), cache = True)
def make_move(board, move, player_id):
    
    player1, player2 = board
    
    selected_square = uint64(1) << move
    
    if player_id == 1:
        flipped_disks = place_disks(selected_square, player1, player2)
        new_player1_board = player1 | (selected_square | flipped_disks)
        new_player2_board = player2 ^ flipped_disks
        
        return new_player1_board, new_player2_board
    
    flipped_disks = place_disks(selected_square, player2, player1)
    new_player1_board = player1 ^ flipped_disks
    new_player2_board = player2 | (selected_square | flipped_disks)
    
    return new_player1_board, new_player2_board

@njit(int16(int8, UniTuple(uint64, 2), int8[::1]), cache=True)
def find_unstable_disks(player_id, board, opponent_moves):
    """
    Identify how many unstable disks the player has.

    Parameters:
    player (int8): The player's ID (1 or 2).
    board (tuple(uint64, uint64)): The game board.
    opponent_moves (int8[::1]): List of opponent moves.

    Returns:
    int16: The number of unstable disks
    """
    unstable_disks = uint64(0)
    player_board, _ = get_player_board(board, player_id)
    
    # Simulate opponent moves to identify unstable disks
    for move in opponent_moves:
        simulated_board = make_move(board, move, player_id)
        _, simulated_opponent_board = get_player_board(simulated_board, player_id)
        unstable_disks |= (player_board & simulated_opponent_board)
    
    return count_bits(unstable_disks)

@njit(int16(int8, UniTuple(uint64, 2), uint64), cache=True)
def find_stable_disks(player_id, board, adjacent_cells):
    """
    Identify how many stable disks the player has.

    Parameters:
    board (tuple(uint64, uint64)): The game board.
    player (int16): The player's ID (1 or 2).
    adjacent_cells (uint64): Bitboard representing adjacent cells to the player's disks.

    Returns:
    int16: The number of stable disks.
    """
    fliped_disks = uint64(0)
    player_board, opponent_board = get_player_board(board, player_id)
    
    # Iterate over all possible adjacent cells
    for i in range(64):
        if adjacent_cells & (uint64(1) << i):
            
            tmp_opponent = opponent_board | (adjacent_cells ^ (uint64(1) << i))
            tmp_board = (player_board, tmp_opponent) if player_id == 1 else (tmp_opponent, player_board)
            simulated_board = make_move(tmp_board, i, player_id)
            _, simulated_opponent_board = get_player_board(simulated_board, player_id)
            fliped_disks |= (player_board & simulated_opponent_board)
            
    stable_disks = player_board & ~fliped_disks
    
    return count_bits(stable_disks)


    