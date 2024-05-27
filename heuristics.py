from constants import *
import bitboard_utils as bbu

@njit(int16(int16, int16), cache=True)
def disk_parity_heuristic(player_disks, opponent_disks):
    """
    Calculate the disk parity heuristic.

    The disk parity heuristic measures the difference in the number of disks 
    between the player and the opponent. It is computed as the percentage difference 
    relative to the total number of disks on the board.

    Args:
        player_disks (int16): The number of disks the player has.
        opponent_disks (int16): The number of disks the opponent has.

    Returns:
        int16: The disk parity heuristic value.
    """
    
    disk_parity_heuristic = int16(100 * (player_disks - opponent_disks) / (player_disks + opponent_disks))
    
    return disk_parity_heuristic

@njit(int16(int16[:, :], int16), cache=True)
def disk_parity_heuristic_standalone(board, player_id):
    """
    Calculate the standalone disk parity heuristic.

    This function calculates the disk parity heuristic based on the current state
    of the board. 
    
    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The disk parity heuristic value.
    """
    bitboard = bbu.array_to_bitboard(board)
    bitboard_player, bitboard_opponent = bbu.get_player_board(bitboard, player_id)
    player_disks = bbu.count_ones(bitboard_player)
    opponent_disks = bbu.count_ones(bitboard_opponent)
    
    return disk_parity_heuristic(player_disks, opponent_disks)
        

@njit(int16(int16, int16, int16, int16), cache=True)
def mobility_heuristic(player_moves_a, opponent_moves_a,
                       player_moves_p, opponent_moves_p):
    """
    Calculate the mobility heuristic.

    The mobility heuristic evaluates both actual and potential mobility.
    - Actual mobility refers to the number of legal moves available to the player.
    - Potential mobility refers to the number of potential moves (empty cells adjacent to opponent's disks).

    The heuristic is the average of the actual and potential mobility values, 
    calculated as percentage differences relative to the total number of moves.

    Args:
        player_moves_a (int16): The number of actual moves available to the player.
        opponent_moves_a (int16): The number of actual moves available to the opponent.
        player_moves_p (int16): The number of potential moves for the player.
        opponent_moves_p (int16): The number of potential moves for the opponent.

    Returns:
        int16: The mobility heuristic value.
    """
    
    if(player_moves_a + opponent_moves_a !=0):
        actual_mobility_heuristic = 100 * (player_moves_a - opponent_moves_a)/(player_moves_a + opponent_moves_a)
    else:
        actual_mobility_heuristic = 0
        
    if(player_moves_p + opponent_moves_p !=0):
        potential_mobility_heuristic = 100 * (player_moves_p - opponent_moves_p)/(player_moves_p + opponent_moves_p)
    else:
        potential_mobility_heuristic = 0
        
    mobility_heuristic = (actual_mobility_heuristic + potential_mobility_heuristic)/2
    
    return mobility_heuristic

@njit(int16(int16[:, :], int16), cache=True)
def mobility_heuristic_standalone(board, player_id):
    """
    Calculate the standalone mobility heuristic.

    This function calculates the mobility heuristic based on the current state of the board.

    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The mobility heuristic value.
    """
    opponent_id = 3 - player_id
    bitboard = bbu.array_to_bitboard(board)
    
    player_possible_moves = bbu.get_possible_moves(player_id, bitboard)
    opponent_possible_moves = bbu.get_possible_moves(opponent_id, bitboard)
    
    player_actual_move_nb = player_possible_moves.shape[0]
    opponent_actual_move_nb = opponent_possible_moves.shape[0]
    
    player_adjacent_cells = bbu.find_empty_neighbors_of_player(player_id, bitboard)
    opponent_adjacent_cells = bbu.find_empty_neighbors_of_player(opponent_id, bitboard)
    
    player_potential_move_nb = bbu.count_ones(opponent_adjacent_cells)
    opponent_potential_move_nb = bbu.count_ones(player_adjacent_cells)
    
    return mobility_heuristic(player_moves_a=player_actual_move_nb, 
                              opponent_moves_a=opponent_actual_move_nb, 
                              player_moves_p=player_potential_move_nb, 
                              opponent_moves_p=opponent_potential_move_nb)

@njit(int16(UniTuple(uint64, 2), int16, int16[:, :], int16[:, :]), cache=True)
def corner_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves):
    """
    Calculate the corner control heuristic.

    The corner heuristic evaluates the control of corner squares (A1, A8, H1, H8).
    It considers both the number of corners occupied and the potential to occupy corners 
    (possible moves to corner positions).

    The heuristic assigns a higher weight to actually captured corners and a lower 
    weight to potential corners, calculated as a percentage difference relative to 
    the total corner values.

    Args:
        bitboard (UniTuple(uint64, 2)): The bitboard representation of the board.
        player_id (int16): The ID of the player (1 or 2).
        player_possible_moves (int16[:, :]): The possible moves for the player.
        opponent_possible_moves (int16[:, :]): The possible moves for the opponent.

    Returns:
        int16: The corner control heuristic value.
    """
    
    corners_mask = uint64(0x8100000000000081) # Corners: a1, a8, h1, h8
    
    bitboard_player, bitboard_opponent = bbu.get_player_board(bitboard, player_id)
    
    player_corners = bbu.count_ones(bitboard_player & corners_mask)
    opponent_corners = bbu.count_ones(bitboard_opponent & corners_mask)
    
    player_potential_corners, opponent_potential_corners = 0, 0
    
    # Iterate through the list of possible moves
    for move in player_possible_moves:
        row, col = move
        if (uint64(1) << (8 * row + col)) & corners_mask:
            player_potential_corners += 1
            
    for move in opponent_possible_moves:
        row, col = move
        if (uint64(1) << (8 * row + col)) & corners_mask:
            opponent_potential_corners += 1
    
    corners_captured_weight, potential_corners_weight = 2, 1
    
    player_corner_value = corners_captured_weight * player_corners + potential_corners_weight * player_potential_corners
    opponent_corner_value = corners_captured_weight * opponent_corners + potential_corners_weight * opponent_potential_corners
    
    if player_corner_value + opponent_corner_value != 0:
        # corner_heuristic = 100 * (player_corner_value - opponent_corner_value) / (4*corners_captured_weight)
        corner_heuristic = 100 * (player_corner_value - opponent_corner_value) / (player_corner_value + opponent_corner_value)
    else:
        corner_heuristic = 0
    
    return corner_heuristic

@njit(int16(int16[:, :], int16), cache=True)
def corner_heuristic_standalone(board, player_id):
    """
    Calculate the standalone corner control heuristic based on the current state of the board.

    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The corner control heuristic value.
    """
    opponent_id = 3 - player_id
    bitboard = bbu.array_to_bitboard(board)
    
    player_possible_moves = bbu.get_possible_moves(player_id, bitboard)
    opponent_possible_moves = bbu.get_possible_moves(opponent_id, bitboard)
    
    return corner_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves)

@njit(int16(UniTuple(uint64, 2), int16, int16[:, :], int16[:, :], uint64, uint64), cache=True)
def stability_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves,
                        player_adjacent_cells, opponent_adjacent_cells):
    """
    Calculate the stability heuristic.

    The stability heuristic evaluates the number of stable and unstable disks.
    - Stable disks are those that cannot be flipped for the rest of the game.
    - Unstable disks are those that can be flipped at the next move of the opponent.

    The heuristic measures the difference between stable and unstable disks for 
    the player and the opponent, calculated as a percentage difference relative 
    to the total stability value.

    Args:
        bitboard (UniTuple(uint64, 2)): The bitboard representation of the board.
        player_id (int16): The ID of the player (1 or 2).
        player_possible_moves (int16[:, :]): The possible moves for the player.
        opponent_possible_moves (int16[:, :]): The possible moves for the opponent.
        player_adjacent_cells (uint64): The bitboard of empty cells adjacent to the player's disks.
        opponent_adjacent_cells (uint64): The bitboard of empty cells adjacent to the opponent's disks.

    Returns:
        int16: The stability heuristic value.
    """
    
    opponent_id = 3 - player_id
    
    player_stable_disks = bbu.find_stable_disks(player_id, bitboard, player_adjacent_cells)
    opponent_stable_disks = bbu.find_stable_disks(opponent_id, bitboard, opponent_adjacent_cells)
    
    player_unstable_disks = bbu.find_unstable_disks(player_id, bitboard, opponent_possible_moves)
    opponent_unstable_disks = bbu.find_unstable_disks(opponent_id, bitboard, player_possible_moves)
    
    stable_weight = 2
    unstable_weight = 1
    
    if(player_stable_disks + opponent_stable_disks != 0):
        stable_disk_heuristic = 100 * (player_stable_disks-opponent_stable_disks)/(player_stable_disks+opponent_stable_disks)
    else:
        stable_disk_heuristic = 0
    
    if(player_unstable_disks + opponent_unstable_disks != 0):
        unstable_disk_heuristic = 100 * (opponent_unstable_disks-player_unstable_disks)/(player_unstable_disks+opponent_unstable_disks)
    else:
        unstable_disk_heuristic = 0
        
    stability_heuristic = (stable_weight * stable_disk_heuristic + unstable_weight * unstable_disk_heuristic)/(stable_weight+unstable_weight)
        
    return stability_heuristic

@njit(int16(int16[:, :], int16), cache=True)
def stability_heuristic_standalone(board, player_id):
    """
    Calculate the standalone stability heuristic based on the current state of the board.

    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The stability heuristic value.
    """
    opponent_id = 3 - player_id
    bitboard = bbu.array_to_bitboard(board)
    
    player_adjacent_cells = bbu.find_empty_neighbors_of_player(player_id, bitboard)
    opponent_adjacent_cells = bbu.find_empty_neighbors_of_player(opponent_id, bitboard)
    
    player_possible_moves = bbu.get_possible_moves(player_id, bitboard)
    opponent_possible_moves = bbu.get_possible_moves(opponent_id, bitboard)
    
    return stability_heuristic(bitboard=bitboard, player_id=player_id, 
                               player_possible_moves=player_possible_moves, opponent_possible_moves=opponent_possible_moves, 
                               player_adjacent_cells=player_adjacent_cells, opponent_adjacent_cells=opponent_adjacent_cells)

@njit(int16(int16[:, :], int16), cache = True)
def hybrid_heuristic(board, player_id):
    """
    Evaluate the board state using a hybrid heuristic.

    The hybrid heuristic combines multiple heuristics to evaluate the board state:
    - Disk Parity: Measures the difference in the number of disks.
    - Mobility: Evaluates actual and potential mobility.
    - Corner Control: Assesses control and potential control of corner squares.
    - Stability: Measures the number of stable and unstable disks.

    Each heuristic is weighted and combined to compute a final evaluation score.
    
    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The ID of the player (1 or 2).

    Returns:
        int16: The final hybrid heuristic score.
    """
    opponent_id = 3 - player_id
    bitboard = bbu.array_to_bitboard(board)
    bitboard_player, bitboard_opponent = bbu.get_player_board(bitboard, player_id)
    
    # Number of disks
    player_disks = bbu.count_ones(bitboard_player)
    opponent_disks = bbu.count_ones(bitboard_opponent)
    
    # Possible moves
    player_possible_moves = bbu.get_possible_moves(player_id, bitboard)
    opponent_possible_moves = bbu.get_possible_moves(opponent_id, bitboard)
    
    # Adjacents cells
    player_adjacent_cells = bbu.find_empty_neighbors_of_player(player_id, bitboard)
    opponent_adjacent_cells = bbu.find_empty_neighbors_of_player(opponent_id, bitboard)
    
    player_actual_move_nb = player_possible_moves.shape[0]
    opponent_actual_move_nb = opponent_possible_moves.shape[0]
    
    # Potential moves for each players
    player_potential_move_nb = bbu.count_ones(opponent_adjacent_cells)
    opponent_potential_move_nb = bbu.count_ones(player_adjacent_cells)
    
    # =============== Game Over ===============
    
    if player_actual_move_nb + opponent_actual_move_nb == 0:
        disk_diff = player_disks - opponent_disks
        return 400*disk_diff
    
    # =============== Heuristics Values ===============
    
    disk_parity_heuristic_value = disk_parity_heuristic(player_disks, opponent_disks)

    mobility_heuristic_value = mobility_heuristic(player_moves_a=player_actual_move_nb, 
                                            opponent_moves_a=opponent_actual_move_nb, 
                                            player_moves_p=player_potential_move_nb, 
                                            opponent_moves_p=opponent_potential_move_nb)

    corner_heuristic_value = corner_heuristic(bitboard, player_id, player_possible_moves, opponent_possible_moves)
        
    stability_heuristic_value = stability_heuristic(bitboard=bitboard, player_id=player_id, 
                                        player_possible_moves=player_possible_moves, opponent_possible_moves=opponent_possible_moves, 
                                        player_adjacent_cells=player_adjacent_cells, opponent_adjacent_cells=opponent_adjacent_cells)
    
    # =============== Final Score ===============
    
    disk_parity_weight = (player_disks+opponent_disks)/2
    mobility_weight = 20
    corner_weight = 50
    stability_weight = 40
    
    final_score = disk_parity_weight*disk_parity_heuristic_value + mobility_weight*mobility_heuristic_value \
                + corner_weight*corner_heuristic_value + stability_weight*stability_heuristic_value
                
    final_score /= (disk_parity_weight + mobility_weight + corner_weight + stability_weight)
    
    return int16(final_score)

STATIC_WEIGHTS = np.array([
    [ 4, -3,  2,  2,  2,  2, -3,  4],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 2, -1,  1,  0,  0,  1, -1,  2],
    [ 2, -1,  0,  1,  1,  0, -1,  2],
    [ 2, -1,  0,  1,  1,  0, -1,  2],
    [ 2, -1,  1,  0,  0,  1, -1,  2],
    [-3, -4, -1, -1, -1, -1, -4, -3],
    [ 4, -3,  2,  2,  2,  2, -3,  4]
], dtype=np.int16)

@njit(int16(int16[:, :], int16), cache=True)
def static_weights_heuristic(board, player_id):
    """
    Evaluates the board state with Static Weights Heuristic.

    This heuristic evaluates the board state by performing matrix multiplication between the board state
    matrix and the weights matrix. It sums up the products to obtain the final evaluation score.

    Args:
        board (int16[:, :]): The current state of the board.
        player_id (int16): The player's ID (1 or 2).

    Returns:
        int16: The final score of the weighted piece value heuristic.
    """
    # Convert the board to a boolean mask where True represents the player's pieces
    player_mask = (board == player_id)
    opponent_mask = (board == (3 - player_id))
    
    # Perform matrix multiplication between the board state and the weights, then sum up the products
    player_score = np.sum(player_mask * STATIC_WEIGHTS)
    opponent_score = np.sum(opponent_mask * STATIC_WEIGHTS)
    
    return player_score - opponent_score