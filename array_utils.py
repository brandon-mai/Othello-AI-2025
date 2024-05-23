from constants import *

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
    return np.count_nonzero(board == player_id) - np.count_nonzero(board == (PLAYER2 if player_id == PLAYER1 else PLAYER1))

@njit(boolean(int16, int16, UniTuple(int16, 2), int16, int16[:, :]), cache=True)
def check_line(row, col, direction, player, board):
    """
    Check if a line starting from (row, col) in a given direction encloses opponent's tiles and ends at player's tile.

    Parameters:
    row (int16): Starting row.
    col (int16): Starting column.
    direction (UniTuple(int16, 2)): Direction vector (dr, dc).
    player (int16): Current player's ID (1 or 2).
    board (int16[:, :]): The game board.

    Returns:
    boolean: True if a valid enclosing line is found, False otherwise.
    """

    dr, dc = direction
    opponent = 3 - player
    # Move to the next cell in the given direction
    r, c = row + dr, col + dc  
    opponent_tiles = 0
    
    # Traverse the board in the given direction while opponent's tiles are found
    while (0 <= r < 8 and 0 <= c < 8) and board[r, c] == opponent:
        r, c = r + dr, c + dc
        opponent_tiles += 1
    
    # Check if the line ends at player's tile and encloses opponent's tiles
    if 0 <= r < 8 and 0 <= c < 8 and board[r, c] == player and opponent_tiles > 0:
        return True
            
    return False
    
@njit(boolean(UniTuple(int16, 2), int16, int16[:, :]), cache = True)
def is_valid_move(move, player, board):
    """
    Check if a move is valid for the given player on the board.

    Parameters:
    move (UniTuple(int16, 2)): The move position (row, col).
    player (int16): Current player's ID (1 or 2).
    board (int16[:, :]): The game board.

    Returns:
    boolean: True if the move is valid, False otherwise.
    """
    row, col = move
    
    # Move is invalid if the cell is not empty
    if board[row, col] != FREE_CELL:
        return False
    
    # Check all directions for a valid enclosing line
    for dr, dc in DIRECTIONS:
        if check_line(row, col, (dr, dc), player, board):
            return True
    return False

@njit(int16[:, :](int16, int16[:, :]), cache=True)
def get_possible_moves(player, board):
    """
    Get all possible valid moves for the given player on the board.

    Parameters:
    player (int16): Current player's ID (1 or 2).
    board (int16[:, :]): The game board.

    Returns:
    int16[:, :]: Array of valid move positions.
    """
    moves = np.zeros((50, 2), dtype=np.int16)  # Pre-allocate moves array
    move_count = 0
    
    # Iterate over all board cells
    for r in range(8):
        for c in range(8):
            # Check if the cell is empty and the move is valid
            if board[r, c] == FREE_CELL and is_valid_move((r, c), player, board):
                moves[move_count] = (int16(r), int16(c))
                move_count += 1
    
    # Return only the valid moves            
    return moves[:move_count] 

@njit(void(UniTuple(int16, 2), int16, int16[:, :]), cache = True)
def flip_tiles(move, player, board):
    """
    Flip the opponent's tiles to the player's tiles for a given move on the board.

    Parameters:
    move (UniTuple(int16, 2)): The move position (row, col).
    player (int16): Current player's ID (1 or 2).
    board (int16[:, :]): The game board.
    """
    r, c = move
    board[r, c] = player
    
    # Check all directions for valid enclosing lines and flip opponent's tiles
    for dr, dc in DIRECTIONS:
        if check_line(r, c, (dr, dc), player, board):
            # Move to the next cell in the given direction
            rr, cc = r + dr, c + dc 
            # Flip opponent's tiles until the player's tile is found
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] != player:
                board[rr, cc] = player
                rr, cc = rr + dr, cc + dc