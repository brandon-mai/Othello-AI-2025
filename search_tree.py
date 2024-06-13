import time
import numpy as np
from numba import njit, int32, int8, uint64, void, int16, float64, boolean, prange
from numba.types import UniTuple
from numba.experimental import jitclass, structref
from numba.extending import overload, overload_method
from numba import types

from bitboard_utils import get_moves_index, place_disks, possible_moves, count_bits, make_move
from constants import MCTS_BATCH_SIZE

MAX_NODES = 5000000

class SearchTree(structref.StructRefProxy):
    """
    Proxy class representing a search tree for the Monte Carlo Tree Search (MCTS) algorithm.
    This class only stores all the data used to perform the MCTS.

    Attributes:
    - nodes_count (int32): Total number of nodes in the tree.
    - root_id (int32): ID of the root node.
    - parent (np.ndarray): Array storing parent node IDs for each node.
    - first_child (np.ndarray): Array storing the ID of the first child node for each node.
    - num_children (np.ndarray): Array storing the number of children for each node.
    - moves (np.ndarray): Array storing moves associated with each node.
    - player_boards (np.ndarray): Array storing bitboards of player positions for each node.
    - opponent_boards (np.ndarray): Array storing bitboards of opponent positions for each node.
    - num_visits (np.ndarray): Array storing the number of visits for each node.
    - reward (np.ndarray): Array storing the accumulated reward for each node.
    
    Note:
        player_boards and opponent_board are always stored from the current player perspective.
        This means that boards are inverted at every depth of the tree.
        
        player_board always store the board state of the current player and vice-versa
    """
    
    def __new__(cls):
        self = search_tree_ctor()
        return self
    
    @property
    def nodes_count(self):
        return get_nodes_count(self)
    
    @property
    def root_id(self):
        return get_root_id(self)
    
    @property
    def parent(self):
        return get_parent(self)
    
    @property
    def first_child(self):
        return get_first_child(self)
    
    @property
    def num_children(self):
        return get_num_children(self)
    
    @property
    def moves(self):
        return get_moves(self)
    
    @property
    def player_boards(self):
        return get_player_boards(self)
    
    @property
    def opponent_boards(self):
        return get_opponent_boards(self)
    
    @property
    def num_visits(self):
        return get_num_visits(self)
    
    @property
    def reward(self):
        return get_reward(self)

@njit(cache=True)
def get_nodes_count(tree: 'SearchTree') -> int:
    return tree.nodes_count

@njit(cache=True)
def get_root_id(tree: 'SearchTree') -> int:
    return tree.root_id

@njit(cache=True)
def get_parent(tree: 'SearchTree') -> np.ndarray:
    return tree.parent

@njit(cache=True)
def get_first_child(tree: 'SearchTree') -> np.ndarray:
    return tree.first_child

@njit(cache=True)
def get_num_children(tree: 'SearchTree') -> np.ndarray:
    return tree.num_children

@njit(cache=True)
def get_moves(tree: 'SearchTree') -> np.ndarray:
    return tree.moves

@njit(cache=True)
def get_player_boards(tree: 'SearchTree') -> np.ndarray:
    return tree.player_boards

@njit(cache=True)
def get_opponent_boards(tree: 'SearchTree') -> np.ndarray:
    return tree.opponent_boards

@njit(cache=True)
def get_num_visits(tree: 'SearchTree') -> np.ndarray:
    return tree.num_visits

@njit(cache=True)
def get_reward(tree: 'SearchTree') -> np.ndarray:
    return tree.reward

search_tree_fields = [
    ('nodes_count', int32),
    ('root_id', int32),
    ('parent', int32[::1]),
    ('first_child', int32[::1]),
    ('num_children', int32[::1]),
    ('moves', int8[::1]),
    ('player_boards', uint64[::1]),
    ('opponent_boards', uint64[::1]),
    ('num_visits', int32[::1]),
    ('reward', int32[::1])
]

@structref.register
class SearchTreeTypeTemplate(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
structref.define_boxing(SearchTreeTypeTemplate, SearchTree)    
SearchTreeType = SearchTreeTypeTemplate(search_tree_fields)

@njit(void(SearchTreeType), cache=True)
def reset(tree):
    """
    Resets the search tree to initial state.

    Parameters:
        tree (SearchTree): The search tree instance to reset.
    """
    tree.nodes_count = 1
    tree.root_id = 0
    
    tree.parent[0] = -1
    tree.first_child[0] = -1
    tree.num_children[0] = -1
    tree.num_visits[0] = 0
    tree.reward[0] = 0 

@njit(int32(SearchTreeType, uint64, uint64), cache=True)
def define_root(tree, player_board, opponent_board):
    """
    Defines the root node of the search tree with initial player and opponent bitboards.

    Parameters:
        tree (SearchTree): The search tree instance.
        player_board (uint64): Bitboard representing the current player's disks.
        opponent_board (uint64): Bitboard representing the opponent's disks.

    Returns:
        int32: ID of the root node.
    """
    reset(tree)
    tree.player_boards[tree.root_id] = player_board
    tree.opponent_boards[tree.root_id] = opponent_board
    return tree.root_id

@njit(boolean(SearchTreeType, int32), cache = True)
def parent_skiped(tree, node_id):
    """
    Checks if the parent node's skipped his turn.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the current node.

    Returns:
        boolean: True if the parent node's skipped his turn, False otherwise.
    """
    parent_id = tree.parent[node_id]
    if parent_id == -1:
        return False
    all_disks = tree.player_boards[node_id] | tree.opponent_boards[node_id]
    all_parent_disks = tree.player_boards[parent_id] | tree.opponent_boards[parent_id]
    return all_disks == all_parent_disks

@njit(boolean(SearchTreeType, int32), cache = True)
def is_terminal(tree, node_id):
    """
    Checks if the node is terminal (i.e., has no children).

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to check.

    Returns:
        boolean: True if the node is terminal (has no children), False otherwise.
    """
    return tree.num_children[node_id] == 0

@njit(boolean(SearchTreeType, int32), cache = True)
def is_fully_expanded(tree, node_id):
    """
    Checks if all children of the node have been visited at least once.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to check.

    Returns:
        boolean: True if all children are fully expanded, False otherwise.
    """
    first_child = tree.first_child[node_id]
    if first_child == -1:
        return False
    
    num_children = tree.num_children[node_id]
    
    for i in range(num_children):
        if tree.num_visits[first_child + i] <= 0:
            return False
        
    return True

@njit(UniTuple(uint64, 2)(SearchTreeType, int32, int8), cache = True)      
def compute_boards(tree, node_id, move):
    """
    Computes new player and opponent bitboards after making a move.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node.
        move (int8): Move to make.

    Returns:
        UniTuple(uint64, 2): Tuple containing new player bitboard and new opponent bitboard.
    """
    if move == -1:
       
        return tree.player_boards[node_id], tree.opponent_boards[node_id]
    else:
        player_bb = tree.player_boards[node_id]
        opponent_bb = tree.opponent_boards[node_id]
        
        selected_square = uint64(1) << move
        flipped_disks = place_disks(selected_square, player_bb, opponent_bb)
        new_player_board = player_bb | (selected_square | flipped_disks)
        new_opponent_board = opponent_bb ^ flipped_disks
        
    return new_player_board, new_opponent_board

@njit(int32(SearchTreeType, int32), cache = True)
def expand(tree, node_id):
    """
    Expands a node by adding child nodes corresponding to possible moves.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to expand.

    Returns:
        int32: ID of the newly expanded child node.
        
    Note:
        Skipped turns are handled in this function. If the current player has no possible moves and his parent
        didn't already skipped his turn, then the current player is given 1 child which is a dummy move (ie:  -1).
        Otherwise, if the parent has already skipped his turn, the current node is set as terminal.
        
    """
    if tree.first_child[node_id] != -1:
        # If children are already initialized, return a randomly selected unvisited child
        first_child_id = tree.first_child[node_id]
        num_children = tree.num_children[node_id]
        children = slice(first_child_id, first_child_id + num_children)
        unvisited_children = np.where(tree.num_visits[children] == 0)[0]
        move_index = np.random.choice(unvisited_children)
        new_node_id = first_child_id + move_index
        
        move = tree.moves[new_node_id]
        tree.opponent_boards[new_node_id], tree.player_boards[new_node_id] = compute_boards(tree, node_id, move)
        
        return new_node_id
    
    empty_squares = (tree.player_boards[node_id] | tree.opponent_boards[node_id]) ^ 0xFFFFFFFFFFFFFFFF
    possible_moves_bb = possible_moves(tree.player_boards[node_id], tree.opponent_boards[node_id], empty_squares)
    moves = get_moves_index(possible_moves_bb)
    
    nb_moves = moves.shape[0]
    if nb_moves == 0 and not parent_skiped(tree, node_id):
        nb_moves = 1
        moves = np.array([-1], dtype=np.int8)
    elif nb_moves == 0:
        tree.num_children[node_id] = 0
        return node_id
        
    if tree.nodes_count + nb_moves > MAX_NODES:
        print("Max Nodes Reached")
        raise Exception(f'The tree reached the maximum number of nodes authorized -> {MAX_NODES}')
    
    first_child_id = tree.nodes_count
    tree.nodes_count += nb_moves
    tree.first_child[node_id] = first_child_id
    tree.num_children[node_id] = nb_moves
    
    for i in range(nb_moves):
        idx = first_child_id + i
        tree.moves[idx] = moves[i]
        tree.parent[idx] = node_id
        tree.first_child[idx] = -1
        tree.num_children[idx] = -1
        tree.num_visits[idx] = 0
        tree.reward[idx] = 0
    
    move_index = np.random.choice(nb_moves)
    move = moves[move_index]
    new_node_id = first_child_id + move_index
    
    tree.opponent_boards[new_node_id], tree.player_boards[new_node_id] = compute_boards(tree, node_id, move)
    
    return new_node_id

@njit(int32(SearchTreeType, int32, float64), cache=True, fastmath=True)
def best_child(tree, node_id, c_param=1.4):
    """
    Returns the ID of the best child node based on UCB1 score.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to select the best child from.
        c_param (float64): Exploration parameter for UCB1 calculation (default is 1.4).

    Returns:
        int32: ID of the best child node.
    """
    first_child = tree.first_child[node_id]
    num_children = tree.num_children[node_id]
    
    log_total_visits = np.log(tree.num_visits[node_id])
    sqrt_log_total_visits = np.sqrt(2 * log_total_visits)
    
    best_score = -99999
    best_child_index = -1
    for i in range(num_children):
        child_idx = first_child + i
        visits = tree.num_visits[child_idx] + 1e-10
        score = tree.reward[child_idx] / visits + c_param * sqrt_log_total_visits / np.sqrt(visits)
        if score > best_score:
            best_score = score
            best_child_index = i
    
    return int32(first_child + best_child_index)

@njit(int32(SearchTreeType, int32, float64), cache = True)
def tree_policy(tree, node_id, c_param):
    """
    Implements the tree policy for selecting child nodes based on UCB1 scores.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the current node.
        c_param (float64): Exploration parameter for UCB1 calculation.

    Returns:
        int32: ID of the selected child node.
    """
    while not is_terminal(tree, node_id):
        if not is_fully_expanded(tree, node_id):
            return expand(tree, node_id)
        else:
            node_id = best_child(tree, node_id, c_param=c_param)
    return node_id

@njit(int8(SearchTreeType, int32), cache = True)
def default_policy(tree, node_id):
    """
    Simulates a random play-out from a given node until the end of the game.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to start the simulation from.

    Returns:
        int8: Outcome of the simulation (1 for win, -1 for loss).
        
    Note:
        The win or loss is from the POV of the current node. This is important for the backup part.
    """
    
    current_player_bb = tree.player_boards[node_id]
    opponent_bb = tree.opponent_boards[node_id]
    initial_player_color = 1
    pass_count = 0
    
    while pass_count < 2:
        empty_squares = (current_player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb = possible_moves(current_player_bb, opponent_bb, empty_squares)
        valid_moves = get_moves_index(possible_moves_bb)
        if valid_moves.shape[0] == 0:
            pass_count += 1
            current_player_bb, opponent_bb = opponent_bb, current_player_bb
            initial_player_color = -initial_player_color
            continue
        
        pass_count = 0
        move = np.random.choice(valid_moves)
        
        selected_square = uint64(1) << move
        flipped_disks = place_disks(selected_square, current_player_bb, opponent_bb)
        new_player_board = current_player_bb | (selected_square | flipped_disks)
        new_opponent_board = opponent_bb ^ flipped_disks
        
        current_player_bb, opponent_bb = new_opponent_board, new_player_board
        initial_player_color = -initial_player_color
        
    if count_bits(current_player_bb) > count_bits(opponent_bb):
        return initial_player_color
    return -initial_player_color

@njit(void(SearchTreeType, int32, int32), cache = True)
def backup(tree, node_id, reward):
    """
    Backpropagates the result of a simulation up the tree.

    Parameters:
        tree (SearchTree): The search tree instance.
        node_id (int32): ID of the node to start the backpropagation from.
        reward (int32): Reward from the simulation (-1 for loss, 1 for win).
    """
    while node_id != -1:
        tree.num_visits[node_id] += 1
        if reward == 1:
            tree.reward[node_id] += reward
        reward = -reward
        node_id = tree.parent[node_id]
        
@njit(void(SearchTreeType, int32, int32, float64), parallel = True, cache = True)
def search_batch(tree, root, nb_rollouts, c_param):
    """
    Performs batched Monte Carlo Tree Search (MCTS) on a given search tree starting from a specified root node.

    Parameters:
        tree (SearchTreeType): The search tree instance to use for MCTS operations.
        root (int32): The node ID of the root in the search tree.
        nb_rollouts (int32): Number of rollouts (simulations) per batch operation.
        c_param (float64): Exploration parameter for the UCB1 formula in MCTS.
    """
    for _ in range(MCTS_BATCH_SIZE):
        selected_node = tree_policy(tree, root, c_param)
        for _ in prange(nb_rollouts):
            reward = default_policy(tree, selected_node)
            backup(tree, selected_node, -reward)

@njit(int8(SearchTreeType, uint64, uint64, int32, int32, float64), parallel = True, cache = True)
def search(tree, current_player_board, opponent_board, nb_iterations, nb_rollouts, c_param):
    """
    Performs Monte Carlo Tree Search (MCTS) on a given game state represented by the current player's board
    and the opponent's board.

    Parameters:
        tree (SearchTreeType): The search tree instance to use for MCTS operations.
        current_player_board (uint64): Bitboard representing the current player's positions.
        opponent_board (uint64): Bitboard representing the opponent's positions.
        nb_iterations (int32): Number of iterations to run the MCTS algorithm.
        nb_rollouts (int32): Number of rollouts (simulations) per iteration.
        c_param (float64): Exploration parameter for the UCB1 formula in MCTS.

    Returns:
        int8: The best move determined by MCTS.
        
    Note:
        The reward from the Default Policy is inverted because we want it from the POV of the parent.
        This is because the result should reflect how good was the move that led to that Node (ie: The node on which the simulation was done)
    """
    root = define_root(tree, current_player_board, opponent_board)
    
    for _ in range(nb_iterations):
        # Step 1: Selection
        selected_node = tree_policy(tree, root, c_param)
        
        if nb_rollouts > 1:
            for _ in prange(nb_rollouts):
                reward = default_policy(tree, selected_node)
                backup(tree, selected_node, -reward)
        else:
            # Step 2: Simulation
            reward = default_policy(tree, selected_node)
                
            # Step 3: Back propagation
            backup(tree, selected_node, -reward)
        
    return tree.moves[best_child(tree, root, c_param=0)]

# Allows to use SearchTree as a constructor in a jitted function
@njit(SearchTreeType(), cache=True)
def search_tree_ctor():
    st = structref.new(SearchTreeType)
    # Initialize the struct fields
    st.nodes_count = 0
    st.root_id = 0
    
    st.parent = -np.ones(MAX_NODES, dtype=np.int32)
    st.first_child = -np.ones(MAX_NODES, dtype=np.int32)
    st.num_children = -np.ones(MAX_NODES, dtype=np.int32)
    
    st.moves = -np.ones(MAX_NODES, dtype=np.int8)
    st.player_boards = np.zeros(MAX_NODES, dtype=np.uint64)
    st.opponent_boards = np.zeros(MAX_NODES, dtype=np.uint64)
    
    st.num_visits = -np.ones(MAX_NODES, dtype=np.int32)
    st.reward = -np.ones(MAX_NODES, dtype=np.int32)
    
    return st

@overload(SearchTree)
def overload_SearchTree():
    def impl():
        return search_tree_ctor()
    return impl