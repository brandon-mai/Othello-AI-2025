from constants import *

@njit(int16(int16[:, :], int16), cache = True)
def evaluate_disc_parity(board, player_id):
    """
    Simple evaluation of a board calculated as the difference between the number of player's tiles and opponent's tiles.

    Parameters:
    board (int16[:, :]): The current game board.
    player_id (int16): The player's ID (1 or 2).

    Returns:
    int16: The evaluation score.
    """
    player_disks = np.count_nonzero(board == player_id)
    opponent_disks = np.count_nonzero(board == (PLAYER2 if player_id == PLAYER1 else PLAYER1))
    
    # if player_disks == 0: return -9999
    # if opponent_disks == 0: return 9999
    # if player_disks+opponent_disks == 64:
    #     if player_disks > opponent_disks: return 9999
    #     if player_disks < opponent_disks: return -9999
    
    return np.count_nonzero(board == player_id) - np.count_nonzero(board == (PLAYER2 if player_id == PLAYER1 else PLAYER1))

@njit(int16(int16[:, :], int16), cache = True)
def evaluate_mobility(board, player_id):
    """
    Simple evaluation of a board calculated as the difference between the number of player's tiles and opponent's tiles.

    Parameters:
    board (int16[:, :]): The current game board.
    player_id (int16): The player's ID (1 or 2).

    Returns:
    int16: The evaluation score.
    """
    return np.count_nonzero(board == player_id) - np.count_nonzero(board == (PLAYER2 if player_id == PLAYER1 else PLAYER1))