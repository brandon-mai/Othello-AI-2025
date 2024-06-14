from abc import ABC, abstractmethod
from os import environ
import random
import time

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pygame import MOUSEBUTTONDOWN

from heuristics import HEURISTICS
from bitboard_utils import get_moves_index, possible_moves, get_player_board
from constants import CELL_SIZE, INT16_NEGINF, INT16_POSINF, MCTS_BATCH_SIZE
import minmax
import search_tree

class Player(ABC):
    """
    Abstract base class for a player in the Othello game.

    Attributes:
        id (int): The identifier for the player, either 1 or 2.
    """
    def __init__(self):
        self.id = None
    
    def __repr__(self):
        """
        Returns a string representation of the Player object.

        Returns:
            str: A string representation including the player's class name and ID.
        """
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id})"
        
    def set_id(self, player_id):
        """
        Sets the player's ID.

        Parameters:
            player_id (int): The ID to be set for the player.
        """
        if player_id != 1 and player_id != 2:
            raise ValueError("Player ID must be either 1 or 2.")
        
        self.id = player_id

    @abstractmethod
    def get_move(self, board, events):
        """
        Abstract method to be implemented by subclasses to get the player's move.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        raise NotImplementedError('You must implement get_move()')
    
class Agent(Player):
    """
    Class for a AI Agent in the Othello game.
    """
    @abstractmethod
    def copy(self):
        """
        Returns a new instance of the Player with the same parameters as the current instance.
        Used for simulations.

        Returns:
        Agent: A new instance with the same parameters.
        """
        raise NotImplementedError('You must implement copy()') 
    
    
class HumanPlayer(Player):
    """
    Class for a human player in the Othello game.
    """
    def get_move(self, board, events):
        """
        Gets the move from the human player based on mouse input.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        
        player_bb, opponent_bb = get_player_board(board, self.id)
        
        empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        valid_moves = get_moves_index(possible_moves_bb_player)
        
        for event in events:
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                
                bb_indice = row * 8 + col
                if bb_indice in valid_moves:
                    return bb_indice
        return None
    
class RandomAgent(Agent):
    """
    Class for a Random Agent in the Othello game.
    """
    def __init__(self, verbose=False):
               
        self.verbose = verbose
    
    def get_move(self, board, events):
        """
        Gets a random move from the possible moves.

        Parameters:
            board (UniTuple(uint64, 2)): The current state of the game board.
            events (list of pygame.event.Event): A list of Pygame events.

        Returns:
            int: The chosen move as a bitboard index, or None if no valid move is chosen.
        """
        player_bb, opponent_bb = get_player_board(board, self.id)
        
        empty_squares = (player_bb | opponent_bb) ^ 0xFFFFFFFFFFFFFFFF
        possible_moves_bb_player = possible_moves(player_bb, opponent_bb, empty_squares)
        valid_moves = get_moves_index(possible_moves_bb_player)
        move = None
        
        if valid_moves.shape[0] > 0:
            move = random.choice(valid_moves)
            if self.verbose == True:
                print(f"Player {self.id} --> {move}")
            
        return move
    
    def copy(self):
        """
        Returns a new instance of RandomAgent.

        Returns:
        RandomAgent: A new instance with the same parameters.
        """
        return RandomAgent()
    
class MinmaxAgent(Agent):
    """
    Class for a MinMax Agent in the Othello game.
    """
    def __init__(self, depth = 5, time_limit = None, heuristic=HEURISTICS.HYBRID, verbose=False):
        """
        Initializes the MinimaxAgent with a specified search depth.

        Parameters:
            depth (int): The depth to which the Minimax algorithm will search.
            time_limit (float or int): If defined, Iterative Deepening will be applied with this as time constraint.
            verbose (bool): If True, prints debug information during search.
        """

        self.depth = depth
        self.time_limit = time_limit
        self.heuristic = heuristic
        self.verbose = verbose
        self.bot = None
    
    def get_move(self, board, events):
        """
        Determines the best move for the player using the Minimax algorithm.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.
            events (list): A list of game events.

        Returns:
            int: The best move for the player.
        """
        if self.bot is None:
            self.bot = minmax.Minmax(self.id, self.heuristic)
        
        if self.time_limit:
            best_score, best_move = self.iterative_deepening_timed(board)
        else:
            best_score, best_move = self.bot.negamax(board, self.depth, INT16_NEGINF, INT16_POSINF, 1)
            
            if self.verbose:
                print(f"Player {self.id} --> {best_move}/{best_score:<6}")  

        return best_move
    
    def iterative_deepening_timed(self, board):
        """
        Adds the Iterative Deepening Process to the Negamax algorithm with time constraint.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
        """
        
        start_time = time.perf_counter()
        best_move = None
        best_score = 0
        reached_depth = -1
        times = []
        
        for depth in range(1, 20):
            
            iteration_start_time = time.perf_counter()
            
            best_score, best_move = self.mtdf(board, best_score, depth)
            reached_depth = depth
            
            iteration_end_time = time.perf_counter()
            iteration_time = iteration_end_time - iteration_start_time
            times.append(iteration_time)
            
            elapsed_time = iteration_end_time - start_time
            remaining_time = self.time_limit - elapsed_time
            
            # Copmute the geometric progression to estimate the time for the next iteration based on the times of previous iterations.
            if len(times) > 1:
                ratios = [times[i] / times[i - 1] for i in range(1, len(times))]
                avg_ratio = sum(ratios) / len(ratios)
            else:
                avg_ratio = 2  # Assume an initial doubling if we don't have enough data yet
            
            estimated_next_iteration_time = times[-1] * avg_ratio
            
            if remaining_time < estimated_next_iteration_time + 0.1:
                break
            
        if self.verbose:
            print(f"Player {self.id} --> {best_move}/{best_score:<6} (time: {time.perf_counter() - start_time:<5.2f}, reached depth: {reached_depth:<2})")  
              
        return best_score, best_move
    
    def mtdf(self, board, f, depth):
        """
        Implements the MTD(f) algorithm to determine the best move.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.
            f (int16): The first guess for the best value.
            depth (int16): The maximum search depth.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
        """
        g = f
        best_move = -1
        upperBound = INT16_POSINF
        lowerBound = INT16_NEGINF

        while lowerBound < upperBound:
            beta = max(g, lowerBound + 1)
            g, move = self.bot.negamax(board, depth, beta - 1, beta, 1)
            if g < beta:
                upperBound = g
            else:
                lowerBound = g
            best_move = move

        return g, best_move
    
    def copy(self):
        """
        Returns a new instance of MinimaxAgent with the same parameters as the current instance.

        Returns:
            MinimaxPlayer: A new instance with the same parameters.
        """
        player_copy = MinmaxAgent(
            depth=self.depth,
            time_limit=self.time_limit,
            heuristic = self.heuristic,
            verbose=self.verbose
        )
        
        player_copy.set_id(self.id)
        
        return player_copy
    
    
class MCTSAgent(Agent):
    """
    Class for a Monte Carlo Tree Search (MCTS) Agent.
    """
    def __init__(self, time_limit = None, nb_iterations = 100000, nb_rollouts = 1, c_param = 1.4, verbose = False):
        """
        Initializes the MCTSAgent with specified parameters.

        Parameters:
            time_limit (float): The time limit for the MCTS search in seconds.
            nb_iterations (int): The number of iterations for the MCTS search.
            nb_rollouts (int): The number of rollouts for the MCTS search.
            c_param (float): The exploration parameter for the MCTS search.
            verbose (bool): If True, prints debug information during search.
        """
        
        self.time_limit = time_limit
        self.nb_iterations = nb_iterations
        self.verbose = verbose
        self.c_param = c_param
        self.nb_rollouts = nb_rollouts
        
        self.tree = None
        
    def copy(self):
        """
        Returns a new instance of MCTSAgent with the same parameters as the current instance.

        Returns:
            MCTSAgent: A new instance with the same parameters.
        """
        player_copy = MCTSAgent(
            time_limit = self.time_limit,
            nb_iterations=self.nb_iterations,
            c_param=self.c_param,
            nb_rollouts=self.nb_rollouts,
            verbose=self.verbose
        )
        
        player_copy.set_id(self.id)
        
        return player_copy
    
    def timed_search(self, board):
        """
        Conducts a timed MCTS search to determine the best move.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.

        Returns:
            int: The best move for the player.
        """
              
        player_board, opponent_board = get_player_board(board, self.id)
        best_move = -1
        nb_loops = 0
        avg_exec_time = 0
        
        root = search_tree.define_root(self.tree, player_board, opponent_board)
        
        global_start_time = time.perf_counter()
        last_loop_end = global_start_time
        
        while last_loop_end-global_start_time < self.time_limit-avg_exec_time:
            search_tree.search_batch(self.tree, root, nb_rollouts=self.nb_rollouts, c_param=self.c_param)
            
            nb_loops += 1
            loop_end = time.perf_counter()
            loop_time = loop_end - last_loop_end
            last_loop_end = loop_end
                        
            if avg_exec_time == 0:
                avg_exec_time = loop_time
            else:
                avg_exec_time = (avg_exec_time*(nb_loops-1) + loop_time)/nb_loops
            
        best_move = self.tree.moves[search_tree.best_child(self.tree, root, c_param=0)]
        
        if self.verbose:
            print(f"Player {self.id} --> move:{best_move:<2} ({time.perf_counter()-global_start_time:>6.2f} sec, {nb_loops*MCTS_BATCH_SIZE:<7} iterations, {self.nb_rollouts} rollouts)")

        return best_move
    
    def get_move(self, board, events):
        """
        Determines the best move for the player using the MCTS algorithm.

        Parameters:
            board (UniTuple(uint64, 2)): The current game board.
            events (list): A list of game events.

        Returns:
            int: The best move for the player.
        """
        
        if self.tree is None:
            self.tree = search_tree.SearchTree()
        
        player_board, opponent_board = get_player_board(board, self.id)
        
        start_time = time.perf_counter()
        if self.time_limit is not None:
            best_move = self.timed_search(board)
        else:
            best_move = search_tree.search(self.tree, player_board, opponent_board, self.nb_iterations, self.nb_rollouts, self.c_param)
            
            if self.verbose:
                print(f"Player {self.id} --> move:{best_move:<2} ({time.perf_counter()-start_time:>6.2f} sec, {self.nb_iterations} iterations, {self.nb_rollouts} rollouts)")

        return best_move