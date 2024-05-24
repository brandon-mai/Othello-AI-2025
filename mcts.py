import random
import numpy as np
from player import Player
from array_utils import *

class MCTSPlayer(Player):
    def get_move(self, board, valid_moves, events):
        player = MCTS(player=1, board=board, iterations=1)
        move = player.get_best_move()
        return move
    
def game_over(board):
    return len(get_possible_moves(1, board)) == 0 and len(get_possible_moves(2, board)) == 0

def evaluate(board, player): 
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
    def __init__(self, player, board, iterations=100):
        self.player = player
        self.board = board
        self.iterations = iterations
        self.root = Node(player=player, board=board)

    def get_best_move(self) -> tuple:
        for _ in range(self.iterations):
            leaf = self.root.selection()
            assert leaf.is_leaf(), "problème le noeud n'est pas une feuille alors qu'il devrait l'être"
            leaf.extension()
            for child in leaf.children: # ici on pourra tester de plus simuler quand on ajoute un enfant 
                reward = child.simulation(self.iterations)
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
                    print("problème, on est passé par là")
                else:
                    assert child is not None, "problème, Child est None"
                    exploration_term = self.c * np.sqrt(2 * np.log(np.sum(self.rate)) / np.sum(child.rate))
                    weights.append(child.rate[0] / np.sum(child.rate) + exploration_term)

            return self.children[np.argmax(weights)].selection()
        

    def extension(self)-> None:
        possible_moves = get_possible_moves(self.player, self.board)
        for move in possible_moves:
            r, c = move
            new_board = np.copy(self.board)
            flip_tiles((r, c), self.player, new_board)
            new_node = Node(player=1 if self.player == 2 else 2, board=new_board, move=move, parent=self)
            self.add_child(new_node)

    

        
    def simulation(self, nb) -> int:
        board = np.copy(self.board)
        player = self.player


        for i in range(nb):
            while not game_over(board):
                possible_moves = get_possible_moves(player, board)
                move = random.choice(possible_moves)
                flip_tiles(move, player, board)
                player = 3 - player
            reward = evaluate(board, self.player)

        return reward


    def backpropagation(self, r) -> None:
        if r > 0:
            self.rate[0] += 1
        elif r == 0:
            self.rate[1] += 1
        else:
            self.rate[2] += 1

        if self.parent is not None:
            self.parent.backpropagation(-r)

# Initialisation du plateau de jeu
# board = np.zeros((8, 8), dtype=np.int16)
# board[3:5, 3:5] = [[2, 1], [1, 2]]

# # Création du joueur MCTS
# playerMCTS = MCTS(player=1, board=board, iterations=10000)
# move = playerMCTS.get_best_move()
# print(move)
