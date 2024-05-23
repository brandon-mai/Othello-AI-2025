from constants import *

SHIFTS = np.array([1, -1, 8, -8, 9, -9, 7, -7], dtype=np.int16)

@njit(UniTuple(uint64, 2)(UniTuple(uint64, 2), int16),cache = True)
def get_player_board(board, player):
    """
    Return the bitboard of the current player and the opponent
    """
    return board[player - 1], board[2 - player]

@njit(int16(uint64), cache=True)
def numberOfSetBits(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


@njit(int16(UniTuple(uint64, 2), int16), cache=True)
def evaluate(board, player):
    """
    Evaluate the current state of the board.

    Parameters:
    board (UniTuple(uint64, 2)): Tuple containing bitboards for player 1 and player 2.

    Returns:
    int16: Evaluation score of the current board state.
    """
    player_board, opponent_board = get_player_board(board, player)
    
    # Evaluate based on the difference in number of pieces
    return numberOfSetBits(player_board) - numberOfSetBits(opponent_board)

@njit(uint64(int16, UniTuple(uint64, 2)), cache=True)
def find_empty_neighbors_of_player(player_id, board):
    """
    Find the empty neighbors adjacent to a specific player's pieces.

    Parameters:
    player_id (int16): The ID of the player.
    board (UniTuple(uint64, 2)): Tuple containing two uint64 variables representing the pieces of both players.

    Returns:
    uint64: Bitboard representing the empty neighbors adjacent to the specified player's pieces.
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
    
    # Iterate over each direction
    for shift in SHIFTS:
        # Dilate the player's pieces to find adjacent cells in the current direction
        dilation_player = (player_board << shift) | (player_board >> shift)
        
        # Find empty cells adjacent to the player's pieces in the current direction
        empty_neighbors_player |= empty_board & dilation_player
    
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
    moves = np.zeros((50, 2), dtype=np.int16)  # Pre-allocate moves array
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
    
    for direction in range(8):
        if check_line(row, col, direction, player, board):
            dr, dc = DIRECTIONS[direction]
            r, c = row + dr, col + dc
            while (r >= 0 and r < 8) and (c >= 0 and c < 8) and (opponent_board & (uint64(1) << (8 * r + c))):
                opponent_board &= ~(uint64(1) << (8 * r + c))
                player_board |= (uint64(1) << (8 * r + c))
                r, c = r + dr, c + dc
    player_board |= (uint64(1) << (8 * row + col))
    
    # Update the correct player's bitboard in the tuple based on the player ID
    if player == 1:
        return player_board, opponent_board
    else:
        return opponent_board, player_board
    


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
def bitboard_to_array(bitboard):
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

