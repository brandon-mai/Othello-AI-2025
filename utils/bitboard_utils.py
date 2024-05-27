from utils.constants import *

SHIFTS = np.array([1, 8, 9, 7], dtype=np.int16)

@njit(UniTuple(uint64, 2)(int16[:, :]), cache = True)
def array_to_bitboard(board):               
    """
    Converts a 2D array representing the game board into two uint64 bitboards, one for each player.

    Parameters:
    board (numpy.ndarray): The 2D array representing the game board.

    Returns:
    tuple: A tuple containing two uint64 variables, each representing the pieces of one player.
    """
    player1_board = uint64(0)
    player2_board = uint64(0)
    for row in range(8):
        for col in range(8):
            if board[row, col] == 1:
                player1_board |= uint64(1) << (8 * row + col)
            elif board[row, col] == 2:
                player2_board |= uint64(1) << (8 * row + col)
    return player1_board, player2_board

@njit(int16[:, :](uint64), cache = True)
def bitboard_to_1d_array(bitboard):
    """
    Converts a uint64 bitboard into a 2D array representation.

    Parameters:
    bitboard (uint64): The input bitboard.

    Returns:
    numpy.ndarray: A 2D array representation of the bitboard.
    """
    array = np.zeros((8, 8), dtype=np.int16)
    for i in range(64):
        row = i // 8
        col = i % 8
        array[row, col] = (bitboard >> i) & 1
    return array

@njit
def bitboard_to_2d_array(bitboard):
    """
    Converts a tuple of uint64 bitboards into a 2D array representation.

    Parameters:
    bitboard (Tuple[uint64, uint64]): The input bitboard tuple.

    Returns:
    numpy.ndarray: A 2D array representation of the combined bitboards.
    """
    array = np.zeros((8, 8), dtype=np.int16)
    player1, player2 = bitboard
    for i in range(64):
        row = i // 8
        col = i % 8
        if (player1 >> i) & 1:
            array[row, col] = 1
        elif (player2 >> i) & 1:
            array[row, col] = 2
    return array

@njit(UniTuple(uint64, 2)(UniTuple(uint64, 2), int16),cache = True)
def get_player_board(board, player):
    """
    Return the bitboard of the current player and the opponent
    """
    return board[player - 1], board[2 - player]

@njit(int16(uint64), cache=True)
def count_ones(x):
    """
    Count the number of set bits (1s) in a 64-bit unsigned integer.

    Parameters:
    - x (uint64): The input 64-bit unsigned integer.

    Returns:
    - int16: The count of set bits (1s) in the input integer.
    """
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

@njit(uint64(uint64), cache=True)
def shift_n(mask):
    """
    Shift the given bitboard northward, removing bits shifted beyond the top edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting northward.
    """
    return (mask >> 8) & uint64(0x00FFFFFFFFFFFFFF)

@njit(uint64(uint64), cache=True)
def shift_s(mask):
    """
    Shift the given bitboard southward, removing bits shifted beyond the bottom edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting southward.
    """
    return (mask << 8) & uint64(0xFFFFFFFFFFFFFF00)

@njit(uint64(uint64), cache=True)
def shift_e(mask):
    """
    Shift the given bitboard eastward, removing bits shifted beyond the right edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting eastward.
    """
    return (mask << 1) & uint64(0xFEFEFEFEFEFEFEFE)

@njit(uint64(uint64), cache=True)
def shift_o(mask):
    """
    Shift the given bitboard westward, removing bits shifted beyond the left edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting westward.
    """
    return (mask >> 1) & uint64(0x7F7F7F7F7F7F7F7F)

@njit(uint64(uint64), cache=True)
def shift_ne(mask):
    """
    Shift the given bitboard northeastward, removing bits shifted beyond the top-right edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting northeastward.
    """
    return (mask >> 7) & uint64(0xFEFEFEFEFEFEFEFE) & uint64(0x00FFFFFFFFFFFFFF)

@njit(uint64(uint64), cache=True)
def shift_no(mask):
    """
    Shift the given bitboard northwestward, removing bits shifted beyond the top-left edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting northwestward.
    """
    return (mask >> 9) & uint64(0x7F7F7F7F7F7F7F7F) & uint64(0x00FFFFFFFFFFFFFF)

@njit(uint64(uint64), cache=True)
def shift_se(mask):
    """
    Shift the given bitboard southeastward, removing bits shifted beyond the bottom-right edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting southeastward.
    """
    return (mask << 9) & uint64(0xFEFEFEFEFEFEFEFE) & uint64(0xFFFFFFFFFFFFFF00)

@njit(uint64(uint64), cache=True)
def shift_so(mask):
    """
    Shift the given bitboard southwestward, removing bits shifted beyond the bottom-left edge.

    Parameters:
    - mask (uint64): The input bitboard representing the game state.

    Returns:
    - uint64: The resulting bitboard after shifting southwestward.
    """
    return (mask << 7) & uint64(0x7F7F7F7F7F7F7F7F) & uint64(0xFFFFFFFFFFFFFF00)

@njit(uint64(int16, UniTuple(uint64, 2)), cache=True)
def find_empty_neighbors_of_player(player_id, board):
    """
    Find the empty neighboring cells of a player on the game board.

    Parameters:
    - player_id (int16): The ID of the player whose empty neighboring cells are to be found.
    - board (UniTuple): A tuple containing two uint64 bitboards representing the game state.

    Returns:
    - uint64: A bitboard representing the empty neighboring cells of the specified player.
    """
    # Get the bitboards of the current and opponent players
    player_board = board[player_id - 1]
    opponent_board = board[2 - player_id]
    
    # Combine both players' bitboards
    all_players_board = player_board | opponent_board  
    
    # Compute the set of empty cells
    empty_board = ~all_players_board
    
    # Initialize empty neighbor bitboard for the specified player
    empty_neighbors_player = uint64(0)
    
    empty_neighbors_player = empty_board & (shift_n(player_board) | shift_s(player_board) | shift_e(player_board) | shift_o(player_board) \
                                        | shift_ne(player_board) | shift_no(player_board) | shift_se(player_board) | shift_so(player_board))
    return empty_neighbors_player

@njit(boolean(int16, int16, int16, int16, UniTuple(uint64, 2)), cache=True)
def check_line(row, col, direction, player, board):
    """
    Check if a line starting from (row, col) in a given direction encloses opponent's tiles and ends at player's tile.

    Parameters:
        row (int16): Starting row.
        col (int16): Starting column.
        direction (Tuple[int16, int16]): Direction vector (dr, dc).
        player (int16): Current player's ID (1 or 2).
        board (Tuple[uint64, uint64]): Tuple containing bitboards for player 1 and player 2.

    Returns:
        boolean: True if a valid enclosing line is found, False otherwise.
    """
    dr, dc = DIRECTIONS[direction]
    player_board, opponent_board = get_player_board(board, player)
    r, c = row + dr, col + dc
    opponent_tiles = 0
    
    # Move to the next cell in the given direction
    while (r >= 0 and r < 8) and (c >= 0 and c < 8) and (opponent_board & (uint64(1) << (8 * r + c))):
        r, c = r + dr, c + dc
        opponent_tiles += 1
    
    # Check if the line ends at player's tile and encloses opponent's tiles
    if (r >= 0 and r < 8) and (c >= 0 and c < 8) and (player_board & (uint64(1) << (8 * r + c)) and opponent_tiles > 0):
        return True
            
    return False
    
@njit(boolean(UniTuple(int16, 2), int16, UniTuple(uint64, 2)), cache=True)
def is_valid_move(move, player, board):
    """
    Check if a move is valid for the given player on the board.

    Parameters:
    move (UniTuple(int16, 2)): The move position (row, col).
    player (int16): Current player's ID (1 or 2).
    board (tuple(uint64, uint64)): The game board.

    Returns:
    boolean: True if the move is valid, False otherwise.
    """
    row, col = move
    
    # Move is invalid if the cell is not empty
    if (board[0] | board[1]) & (1 << (8 * row + col)):
        return False
    
    # Check all directions for a valid enclosing line
    for direction in range(8):
        if check_line(row, col, direction, player, board):
            return True
    return False

@njit(int16[:, :](int16, UniTuple(uint64, 2)), cache=True)
def get_possible_moves(player, board):
    """
    Get all possible valid moves for the given player on the board.

    Parameters:
    player (int16): Current player's ID (1 or 2).
    board (tuple(uint64, uint64)): The game board.

    Returns:
    int16[:, :]: Array of valid move positions.
    """
    # Pre-allocate moves array
    moves = np.zeros((35, 2), dtype=np.int16)
    move_count = 0
    
    # Find empty neighbors adjacent to opponent's pieces
    opponent_id = 3 - player  # Determine the opponent's ID
    empty_neighbors_opponent = find_empty_neighbors_of_player(opponent_id, board)
    
    # Iterate over all empty neighbors adjacent to the opponent's pieces
    for row in range(8):
        for col in range(8):
            if empty_neighbors_opponent & (1 << (8 * row + col)):
                # Check if the move is valid for the current player
                move = (row, col)
                if is_valid_move(move, player, board):
                    moves[move_count] = (int16(row), int16(col))
                    move_count += 1
    
    # Return only the valid moves            
    return moves[:move_count]

@njit(int16(int16, UniTuple(uint64, 2)), cache=True)
def get_possible_moves_nb(player, board):
    """
    Get all possible valid moves for the given player on the board.

    Parameters:
    player (int16): Current player's ID (1 or 2).
    board (tuple(uint64, uint64)): The game board.

    Returns:
    int16[:, :]: Array of valid move positions.
    """
    move_count = 0
    
    # Find empty neighbors adjacent to opponent's pieces
    opponent_id = 3 - player  # Determine the opponent's ID
    empty_neighbors_opponent = find_empty_neighbors_of_player(opponent_id, board)
    
    # Iterate over all empty neighbors adjacent to the opponent's pieces
    for row in range(8):
        for col in range(8):
            if empty_neighbors_opponent & (1 << (8 * row + col)):
                # Check if the move is valid for the current player
                move = (row, col)
                if is_valid_move(move, player, board):
                    move_count += 1
    
    return move_count

@njit(UniTuple(uint64, 2)(UniTuple(int16, 2), int16, UniTuple(uint64, 2)), cache=True)
def flip_tiles(move, player, board):
    """
    Flip the opponent's tiles to the player's tiles for a given move on the board.

    Parameters:
    row (int16): The move row.
    col (int16): The move column.
    player (int16): Current player's ID (1 or 2).
    player_board (int64): Bitboard representing player's tiles.
    opponent_board (int64): Bitboard representing opponent's tiles.
    """
    row, col = move
    player_board, opponent_board = board
    
    # Get the player's and opponent's bitboards based on the player ID
    player_board, opponent_board = get_player_board(board, player)
    player_board |= (uint64(1) << (8 * row + col))
    
    for direction in range(8):
        if check_line(row, col, direction, player, board):
            dr, dc = DIRECTIONS[direction]
            r, c = row + dr, col + dc
            while (r >= 0 and r < 8) and (c >= 0 and c < 8) and (opponent_board & (uint64(1) << (8 * r + c))):
                opponent_board &= ~(uint64(1) << (8 * r + c))
                player_board |= (uint64(1) << (8 * r + c))
                r, c = r + dr, c + dc
    
    # Update the correct player's bitboard in the tuple based on the player ID
    if player == 1:
        return player_board, opponent_board
    else:
        return opponent_board, player_board
    

@njit(int16(int16, UniTuple(uint64, 2), int16[:, :]), cache=True)
def find_unstable_disks(player, board, opponent_moves):
    """
    Identify how many unstable disks the player has.

    Parameters:
    player (int16): The player's ID (1 or 2).
    board (tuple(uint64, uint64)): The game board.
    opponent_moves (int16[:, :]): List of opponent moves.

    Returns:
    int16: The number of unstable disks
    """
    unstable_disks = uint64(0)
    player_board, _ = get_player_board(board, player)
    opponent = 3 - player
    
    # Simulate opponent moves to identify unstable disks
    for move in opponent_moves:
        row, col = move
        simulated_board = flip_tiles((row, col), opponent, board)
        _, simulated_opponent_board = get_player_board(simulated_board, player)
        unstable_disks |= (player_board & simulated_opponent_board)
    
    return count_ones(unstable_disks)

@njit(int16(int16, UniTuple(uint64, 2), uint64), cache=True)
def find_stable_disks(player, board, adjacent_cells):
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
    player_board, opponent_board = get_player_board(board, player)
    opponent = 3 - player
    
    # Iterate over all possible adjacent cells
    for i in range(64):
        if adjacent_cells & (uint64(1) << i):
            row, col = divmod(i, 8)
            
            tmp_opponent = opponent_board | (adjacent_cells ^ (uint64(1) << i))
            tmp_board = (player_board, tmp_opponent) if player == 1 else (tmp_opponent, player_board)
            simulated_board = flip_tiles((row, col), opponent, tmp_board)
            _, simulated_opponent_board = get_player_board(simulated_board, player)
            fliped_disks |= (player_board & simulated_opponent_board)
            
    stable_disks = player_board & ~fliped_disks
    
    return count_ones(stable_disks)