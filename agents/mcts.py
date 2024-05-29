from collections import defaultdict
import time
import numpy as np
from agents import Agent, RandomAgent
from utils.array_utils import get_possible_moves, flip_tiles
from utils.constants import *
from numba import int32



class MCTSNode:
    def __init__(self, board, player_id, parent = None, move = None):
        self.board = board
        self.move = move
        self.player_id = player_id
        self.parent = parent
        self.children = []
        self.visits = 0
        self._rewards = defaultdict(int)
        self._untried_moves = None
        self._opponent_untried_moves = None
        self.skip = False
        
        # 35 is an upper bound for the max number of possible moves
        self.children_visits = np.zeros(35, dtype=np.int32)
        self.children_rewards = np.zeros(35, dtype=np.int32)
        
    @property
    def untried_moves(self):
        if self._untried_moves is None:
            self._untried_moves = get_possible_moves(self.player_id, self.board)
            if self._untried_moves.shape[0] == 0:
                self.skip = True
                
        return self._untried_moves
    
    @untried_moves.setter
    def untried_moves(self, new_list):
        self._untried_moves = new_list
    
    @property
    def opponent_untried_moves(self):
        if self._opponent_untried_moves is None:
            self._opponent_untried_moves = get_possible_moves(3 - self.player_id, self.board)
                
        return self._opponent_untried_moves
    
    @property
    def reward(self):
        wins = self._rewards[self.parent.player_id]
        return wins
        

    def expand(self):
                
        new_board = np.copy(self.board)
        # Player passes his turn
        if self.skip:
            child_node = MCTSNode(new_board, 3 - self.player_id, self, None)
            self.skip = False
        else:
            nb_moves = self._untried_moves.shape[0]
        
            move_index = np.random.randint(nb_moves)
            r, c = self._untried_moves[move_index]
            
            self._untried_moves[move_index:-1] = self._untried_moves[move_index+1:]
            self._untried_moves = self._untried_moves[:-1]
            
            flip_tiles((r, c), self.player_id, new_board)
            child_node = MCTSNode(new_board, 3 - self.player_id, self, (r, c))
        
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0 and not self.skip
    
    @staticmethod
    @njit(int16(int32, int32, int32[:], int32[:], int32), cache = True, nogil = True)
    def best_index(nb_children, node_visits, children_visits, children_rewards, c):
        
        tmp_child_visits = children_visits[:nb_children]
        tmp_child_rewards = children_rewards[:nb_children]
        
        # Calculate UCB1 values
        choices_weights = (tmp_child_rewards / tmp_child_visits) + c * np.sqrt(2 * np.log(node_visits) / tmp_child_visits)

        # Get the index of the best child
        index = np.argmax(choices_weights[:node_visits])
        
        return index

    def best_child(self, c: float = 1.4):
        if not self.children:
            return None
        
        return self.children[self.best_index(len(self.children), self.visits, self.children_visits, self.children_rewards, c)]

    def backup(self, result):
        self.visits += 1
        self._rewards[result] += 1
        if self.parent:
            index_in_parent = self.parent.children.index(self)
            self.parent.children_visits[index_in_parent] += 1
            self.parent.children_rewards[index_in_parent] = self.reward
            self.parent.backup(result)
            
    def is_terminal(self):
        if self.untried_moves.shape[0] == 0:
            if self.opponent_untried_moves.shape[0] == 0:
                return True
        return False

class MCTSAgent(Agent):
    def __init__(self, id, nb_iterations=None, time_limit=None, c=1.4, verbose = False):
        super().__init__(id)
        self.nb_iterations = nb_iterations
        self.time_limit = time_limit
        self.c = c
        self.verbose = verbose
        
        self.root = None
        self.node_count = 0
        self.iteration_count = 0
        

    def get_move(self, board, events):
        move = self.search(board)
        if self.verbose:
            print(f"Player {self.id} --> {move} ({self.node_count:<5} nodes explored | {self.iteration_count:<5} iterations)")  
        return move

    def tree_policy(self):
        current_node = self.root
        self.iteration_count += 1
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                self.node_count += 1
                return current_node.expand()
            else:
                current_node = current_node.best_child(self.c)
        return current_node
    
    def search(self, board):
        
        self.node_count = 0
        self.iteration_count = 0
        self.root = MCTSNode(board, self.id)
        
        if self.nb_iterations is not None:
            for _ in range(self.nb_iterations):            
                node = self.tree_policy()
                result = self.default_policy(node.board, node.player_id)
                node.backup(result)
        else:
            if self.time_limit is None:
                raise Exception('You must either specify the number of iteration of a time limit')

            end_time = time.perf_counter() + self.time_limit
            while True:
                node = self.tree_policy()
                result = self.default_policy(node.board, node.player_id)
                node.backup(result)
                
                if time.perf_counter() > end_time:
                    break
        
        best_node = self.root.best_child(0)
        if best_node is None:
            return None
        
        return best_node.move
    
    @staticmethod
    @njit(int16(int16[:, :], int16), cache = True, nogil = True)
    def default_policy(board, player_id):
        simu_board = np.copy(board)
        current_player = player_id
        pass_count = 0
        
        while pass_count < 2:
            moves = get_possible_moves(current_player, simu_board)
            if moves.shape[0] == 0:
                pass_count += 1
                current_player = 3 - current_player
                continue
            
            pass_count = 0  # Reset pass count if a move is made
            move_index = np.random.randint(moves.shape[0])
            r, c = moves[move_index]
            flip_tiles((r, c), current_player, simu_board)
            current_player = 3 - current_player

        player_disks = np.count_nonzero(simu_board == player_id)
        opponent_disks = np.count_nonzero(simu_board == 3 - player_id)

        if player_disks > opponent_disks:
            return player_id
        elif opponent_disks > player_disks:
            return 3 - player_id
        return 0
    
    def copy(self):
        return MCTSAgent(self.id, self.nb_iterations, self.time_limit, self.c)







