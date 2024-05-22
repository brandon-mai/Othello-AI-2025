import random
from player import Player
import numpy as np
from utils import *


class MCTSPlayer(Player):
    def get_move(self, board, valid_moves, events):
        # Placeholder for MCTS algorithm
        return random.choice(valid_moves) if valid_moves else None
    


def game_over(board):
    return len(get_possible_moves(1, board)) == 0 and len(get_possible_moves(2, board)) == 0

class MCTS:
    def __init__(self, player, board, iterations=1000):
        self.player = player
        self.board = board
        self.iterations = iterations
        self.root = Node(player=player, board=board)    
    
    def get_best_move(self)->tuple:

        for _ in range(self.iterations):
            leaf = self.root.selection()
            assert leaf.is_leaf(), "problème le noeud n'est pas une feuille alors q'il devrait l'être"
            leaf.extension()
            assert not leaf.is_leaf(), "problème le noeud est une feuille alors qu'il ne devrait pas l'être"
            reward = leaf.simulation()
            leaf.backpropagation(reward)


        #on prends le fils avec le meilleur ratio
        winRate = []
        for child in self.root.children:
            winRate.append(child.Rate[0] / np.sum(child.Rate))

        assert winRate, "problème, il n'y a pas de fils"
        chosen_one = np.argmax(winRate)
        return self.root.children[chosen_one].move

# on est le player 2 --> "O" 
class Node:
    def __init__(self, player:int, board=None, move=None,  parent=None, c=1.4):
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

    def is_root(self)->bool :
        return self.parent == None

    def selection(self) -> 'Node':
        if self.is_leaf():
            return self
        else:
            weights = []
            for child in self.children:
                if np.sum(child.rate) == 0:
                    weights.append(100000000000)  # Choose unvisited nodes first
                else:
                    assert child is not None, "problème, Child est None"
                    if self.is_root():
                        exploration_term = 0
                    else:
                        exploration_term = self.c * np.sqrt(2*np.log(np.sum(self.parent.Rate)) / np.sum(child.rate))
                    weights.append(child.rate[0] / np.sum(child.rate) + exploration_term)

            return np.random.choice(self.children, p=weights / np.sum(weights)).selection()


    
    def extension(self)->None:
        possible_moves = get_possible_moves(self.player, self.board)
        for move in possible_moves:
            new_board = np.copy(self.board)
            r, c = move
            flip_tiles((r, c), self.player, new_board)

            child_node = Node(player=1 if self.player == 2 else 2, move=move, board=new_board, parent=self)
            assert child_node != None, "problème, le noeud n'a pas été créé"
            self.add_child(child_node)

    def simulation(self)->int: # [win, draw, loss]
        current_board = np.copy(self.board)
        current_player = self.player
        while not game_over(current_board): # ici il faudra faire une opti pour pas tout recalculer à chaque fois
            possible_moves = get_possible_moves(current_player, current_board)
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            r, c = move
            flip_tiles((r, c), current_player, current_board)
            current_player = 1 if current_player == 2 else 2

        return evaluate(current_board, self.player)
    
    def backpropagation(self, r)->None:
        if r > 0:
            self.rate[0] += 1
        elif r == 0:
            self.rate[1] += 1
        else:
            self.rate[2] += 1

        if self.parent is not None:
            self.parent.backpropagation(-r)
            
            

board = np.zeros((8, 8), dtype=np.int16)
board[3:5, 3:5] = [[2, 1], [1, 2]]

playerMCTS = MCTS(player=1, board=board, iterations=1000)
move = playerMCTS.get_best_move()
print(move)