import time
import numpy as np
from numba import njit, int32, int8, uint64
from numba.experimental import jitclass, structref
from numba.extending import overload, overload_method
from numba import types

from bitboard_helpers import count_bits
from bitboard_utils import get_moves_index, place_disks, possible_moves

MAX_NODES = 2000000

class SearchTree(structref.StructRefProxy):
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

@njit(cache=True)
def reset(tree):
    tree.nodes_count = 1
    tree.root_id = 0
    
    tree.parent[0] = -1
    tree.first_child[0] = -1
    tree.num_children[0] = -1
    tree.num_visits[0] = 0
    tree.reward[0] = 0 

@njit(cache=True)
def define_root(tree, player_board, opponent_board):
    reset(tree)
    tree.player_boards[tree.root_id] = player_board
    tree.opponent_boards[tree.root_id] = opponent_board
    return tree.root_id

@njit(cache = True)
def search(tree, current_player_board, opponent_board, num_simulations):
    root = define_root(tree, current_player_board, opponent_board)
    for _ in range(num_simulations):
        selected_node = tree_policy(tree, root)
        reward = default_policy(tree, selected_node)
        backup(tree, selected_node, -reward)
    
    return tree.moves[best_child(tree, root, c_param=0)]

@njit(cache = True)
def tree_policy(tree, node_id):
    while not is_terminal(tree, node_id):
        if not is_fully_expanded(tree, node_id):
            return expand(tree, node_id)
        else:
            node_id = best_child(tree, node_id, c_param=1.4)
    return node_id

@njit(cache=True, fastmath=True)
def best_child(tree, node_id, c_param=1.4):
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
    
    return first_child + best_child_index

@njit(cache = True)
def default_policy(tree, node_id):
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
        selected_square = np.uint64(1) << np.uint64(move)
        flipped_disks = place_disks(selected_square, opponent_bb)
        new_player_board = current_player_bb | (selected_square | flipped_disks)
        new_opponent_board = opponent_bb ^ flipped_disks
        current_player_bb, opponent_bb = new_opponent_board, new_player_board
        initial_player_color = -initial_player_color
        
    if count_bits(current_player_bb) > count_bits(opponent_bb):
        return initial_player_color
    return -initial_player_color

@njit(cache = True)
def backup(tree, node_id, reward):
    while node_id != -1:
        tree.num_visits[node_id] += 1
        if reward == 1:
            tree.reward[node_id] += reward
        reward = -reward
        node_id = tree.parent[node_id]

@njit(cache = True)      
def compute_boards(tree, node_id, move):
    if move == -1:
        new_player_board = tree.opponent_boards[node_id]
        new_opponent_board = tree.player_boards[node_id]
    else:
        opponent_bb = tree.opponent_boards[node_id]
        selected_square = np.uint64(1) << np.uint64(move)
        flipped_disks = place_disks(selected_square, opponent_bb)
        new_player_board = tree.player_boards[node_id] | (selected_square | flipped_disks)
        new_opponent_board = opponent_bb ^ flipped_disks
        
    return new_player_board, new_opponent_board

@njit(cache = True)
def expand(tree, node_id):
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
    if nb_moves == 0:
        nb_moves = 1
        moves = np.array([-1], dtype=np.int8)
        
    if tree.nodes_count + nb_moves > MAX_NODES:
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

@njit(cache = True)
def parent_skiped(tree, node_id):
    parent_id = tree.parent[node_id]
    if parent_id == -1:
        return False
    all_disks = tree.player_boards[node_id] | tree.opponent_boards[node_id]
    all_parent_disks = tree.player_boards[parent_id] | tree.opponent_boards[parent_id]
    return all_disks == all_parent_disks

@njit(cache = True)
def is_terminal(tree, node_id):
    return tree.num_children[node_id] == 0 and parent_skiped(tree, node_id)

@njit(cache = True)
def is_fully_expanded(tree, node_id):
    first_child = tree.first_child[node_id]
    if first_child == -1:
        return False
    
    num_children = tree.num_children[node_id]
    
    for i in range(num_children):
        if tree.num_visits[first_child + i] <= 0:
            return False
        
    return True

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