import random
import numpy as np
from player import Player
from array_utils import *

class MCTSPlayer(Player):
    def __init__(self, nb = 1000):
        self.iterations = nb
    
    def get_move(self, board, valid_moves, events):
        player = MCTS(player=1, board=board, iterations=self.iterations)
        move = player.get_best_move()
        return move
    
def game_over(board):
    return len(get_possible_moves(1, board)) == 0 and len(get_possible_moves(2, board)) == 0

def evaluate(board, player): # renvoie 1 pour une victoire, 0 pour une égalité et -1 pour une défaite 
    assert game_over(board), "problème, la partie n'est pas terminée"
    player_score = np.sum(board == player)
    opponent_score = np.sum(board == 3 - player)
    if player_score > opponent_score:
        return 1
    elif player_score == opponent_score:
        return 0
    else:
        return -1

class MCTS:
    def __init__(self, player, board, iterations=1):
        self.player = player
        self.board = board
        self.iterations = iterations
        self.root = Node(player=player, board=board)

    def get_best_move(self) -> tuple:
        for _ in range(self.iterations):
            leaf = self.root.selection()
            leaf.extension()
            for child in leaf.children: # ici on pourra tester de plus simuler quand on ajoute un enfant 
                reward = child.simulation()
                child.backpropagation(reward)

        
        win_rate = [child.rate[0] / np.sum(child.rate) for child in self.root.children if np.sum(child.rate) > 0]
        assert win_rate, "problème, il n'y a pas de fils avec des simulations"
        chosen_one = np.argmax(win_rate)
        return self.root.children[chosen_one].move

class Node:
    def __init__(self, player: int, board=None, move=None, parent=None, c=1.4):
        self.move = move
        self.parent = parent
        self.children = []
        self.board = np.copy(board)
        self.rate = [0, 0, 0]  # [win, draw, loss]
        self.c = c
        self.player = player

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def become_root(self) -> None:
        self.parent = None

    def is_root(self) -> bool:
        return self.parent is None

    def selection(self) -> 'Node':
        if self.is_leaf():
            return self
        else:
            weights = []
            for child in self.children:
                if np.sum(child.rate) == 0: # théoriquement on devrait jamais passer par là
                    weights.append(1e12)  # Choose unvisited nodes first
                else:
                    assert child is not None, "problème, Child est None"
                    exploration_term = self.c * np.sqrt(2 * np.log(np.sum(self.rate)) / np.sum(child.rate))
                    weights.append(child.rate[0] / np.sum(child.rate) + exploration_term)

            return self.children[np.argmax(weights)].selection()
        

    def extension(self): 
        if not game_over(self.board): 
            if self.children != []:
                print("It is not a leaf, can not extand")
                return
            else: 
                for move in get_possible_moves(self.player, self.board):
                    new_board = np.copy(self.board).astype(np.int16)
                    # conversion pour numba
                    move = tuple(move.astype(np.int16))
                    self.player = np.int16(self.player)
                    flip_tiles(move, self.player, new_board)
                    new_node = Node(player= 3 - self.player, board=new_board, move=move, parent=self)
                    self.add_child(new_node)


    def simulation(self) -> int: # fais la simulation d'une partie et renvoie le résultat
        temp_board = np.copy(self.board).astype(np.int16)
        player = self.player.astype(np.int16)

        while not game_over(temp_board):
            possible_moves = get_possible_moves(player, temp_board)
            if possible_moves.size == 0:
                player = 3 - player
                continue
            move = tuple(random.choice(possible_moves).astype(np.int16)) # conversion pour numba
            flip_tiles(move, player, temp_board)
            player = 3 - player

        reward = evaluate(temp_board, self.player)
        assert reward in [-1, 0, 1], "problème, le reward n'est pas dans les valeurs attendues"
        return reward
    
    def backpropagation(self, reward) -> None:
        if reward == 1:
            self.rate[0] += 1
        elif reward == 0:
            self.rate[1] += 1
        elif reward == -1:
            self.rate[2] += 1
        else:
            raise ValueError("problème, le reward n'est pas dans les valeurs attendues")
        if self.parent is not None:
            self.parent.backpropagation(reward)

#Initialisation du plateau de jeu
board = np.zeros((8, 8), dtype=np.int16)
board[3:5, 3:5] = [[2, 1], [1, 2]]

# test des fonctions

n = Node(player=1, board=board)
f = n.selection()
print(f, "est bien une feuile ? : ", f.is_leaf())
print(get_possible_moves(1, f.board))
f.extension()
print(f, "est bien une feuile ? : ", f.is_leaf())
print(len(f.children))
r = f.children[0].simulation()
print(r)
f.children[0].backpropagation(r)


# test de la classe MCTS
player = MCTS(player=1, board=board, iterations=10000)
move = player.get_best_move()
print(move)

