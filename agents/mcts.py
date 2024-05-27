import numpy as np
from agents.player import Agent
from agents.random import RandomAgent
from utils.array_utils import get_possible_moves, flip_tiles
from utils.constants import *

class MCTSNode:
    def __init__(self, board, player_id, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.player_id = player_id
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = get_possible_moves(player_id, board)

    def expand(self):
        if self.untried_moves.size == 0:
            return None
        
        r, c = self.untried_moves[-1]
        self.untried_moves = self.untried_moves[:-1]
        
        new_board = np.copy(self.board)
        flip_tiles((r, c), self.player_id, new_board)
        child_node = MCTSNode(new_board, 3 - self.player_id, self, (r, c))
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_value):
        if not self.children:
            return None
        choices_weights = [(child.wins / child.visits) + exploration_value * np.sqrt(2 * np.log(self.visits) / child.visits) for child in self.children]
        return self.children[np.argmax(choices_weights)]

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

@njit(int16(int16[:, :], int16), cache = True)
def random_simulation(board, player_id):
    current_player = player_id
    while True:
        moves = get_possible_moves(current_player, board)
        if moves.size == 0:
            if get_possible_moves(3 - current_player, board).size == 0:
                break
            current_player = 3 - current_player
            continue
        move_index = np.random.randint(len(moves))
        r, c = moves[move_index]
        flip_tiles((r, c), current_player, board)
        current_player = 3 - current_player

    player_disks = np.count_nonzero(board == player_id)
    opponent_disks = np.count_nonzero(board == 3 - player_id)

    if player_disks > opponent_disks:
        return 1
    else:
        return 0

class MCTSAgent(Agent):
    def __init__(self, id, iteration_limit):
        super().__init__(id)
        self.iteration_limit = iteration_limit
        self.root = None
        
    def copy(self):
        return MCTSAgent(self.id, self.iteration_limit)

    def get_move(self, board, events):
        if self.root is None or not np.array_equal(self.root.board, board):
            self.root = MCTSNode(board, self.id)
        else:
            self.root = self.find_new_root(board)

        for _ in range(self.iteration_limit):
            node = self.tree_policy(self.root)
            result = self.default_policy(node.board, node.player_id)
            node.backpropagate(result)
        
        best_node = self.root.best_child(0)
        if best_node is None:
            raise Exception("No valid moves found.")
        return best_node.move
    
    def find_new_root(self, board):
        for child in self.root.children:
            if np.array_equal(child.board, board):
                return child
            
        return MCTSNode(board, self.id)

    def tree_policy(self, node):
        while not self.is_terminal(node):
            if not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node:
                    return expanded_node
            else:
                node = node.best_child(np.sqrt(2))
        return node

    def default_policy(self, board, player_id):
        board_copy = np.copy(board)
        return random_simulation(board_copy, player_id)

    def is_terminal(self, node):
        return node.is_fully_expanded() and all(child.is_fully_expanded() for child in node.children)

# Fonction pour effectuer une partie entre deux joueurs
def play_game(player1, player2):
    board = np.zeros((8, 8), dtype=np.int16)
    board[3:5, 3:5] = [[2, 1], [1, 2]]
    
    current_player = 1  # Joueur 1 commence

    while True:
        valid_moves = get_possible_moves(current_player, board)

        if valid_moves.shape[0] == 0:
            current_player = 3 - current_player  # Passe au joueur suivant
            next_valid_moves = get_possible_moves(current_player, board)
            if not next_valid_moves:  # Aucun joueur ne peut jouer
                break
            continue

        if current_player == 1:
            move = player1.get_move(board, None)
        else:
            move = player2.get_move(board, None)

        if move is not None:
            r, c = move
            flip_tiles((r, c), current_player, board)
            print(current_player, move)
            current_player = 3 - current_player  # Passe au joueur suivant
        else:
            current_player = 3 - current_player  # Passe au joueur suivant si aucun coup valide

    player1_disks = np.count_nonzero(board == 1)
    player2_disks = np.count_nonzero(board == 2)

    if player1_disks > player2_disks:
        print("Player 1 wins!")
    elif player1_disks < player2_disks:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

# Définition des paramètres
iteration_limit = 10000  # Nombre d'itérations pour MCTS

# Création des joueurs
mcts_player = MCTSAgent(1, iteration_limit)
random_player = RandomAgent(2)

# Lancement de la partie
play_game(mcts_player, random_player)




