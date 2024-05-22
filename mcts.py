import random
from player import Player
import numpy as np
from utils import *


class MCTSPlayer(Player):
    def get_move(self, board, valid_moves, events):
        # Placeholder for MCTS algorithm
        return random.choice(valid_moves) if valid_moves else None
    

# on est le player 2 --> "O" 
class Node:
    def __init__(self, player:int, state=None, move=None,  parent=None, c=1.4):
        self.move = move
        self.parent = parent
        self.children = []
        self.state = np.copy(state)
        self.Rate = [0, 0, 0]  # [win, draw, loss]
        self.c = c
        self.player = player

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0
    
    def become_root(self)->None:
        del self.parent
        self.parent = None

    def is_root(self)->bool :
        return self.parent == None

    def selection(self) -> 'Node':
        if self.is_leaf():
            return self
        else:
            weights = []
            for child in self.children:
                if np.sum(child.Rate) == 0:
                    weights.append(100000000000)  # Choose unvisited nodes first
                else:
                    assert child is not None, "problème, Child est None"
                    if self.is_root():
                        exploration_term = 0
                    else:
                        exploration_term = self.c * np.sqrt(np.log(np.sum(self.parent.Rate)) / np.sum(child.Rate))
                    weights.append(child.Rate[0] / np.sum(child.Rate) + exploration_term)

            return np.random.choice(self.children, p=weights / np.sum(weights))


    
    def extension(self)->None:
        possible_moves  = get_possible_moves(self.player, self.state)
        for move in possible_moves:
            new_state = np.copy(self.state)
            flip_tiles(move, self.player, new_state)

            child_node = Node(player='X' if self.player == 'O' else 'O', move=move, state=new_state, parent=self)
            assert child_node != None, "problème, le noeud n'a pas été créé"
            self.add_child(child_node)

    def simulation(self)->int: # [win, draw, loss]
        current_state = np.copy(self.state)
        current_player = self.player
        while not game_over(current_state): #ici il faudra faire une opti pour pas tout recalculer à chaque fois
            possible_moves = get_possible_moves(current_player, current_state)
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            flip_tiles(move, current_player, current_state)
            current_player = 'X' if current_player == 'O' else 'O'

        return evaluate(current_state, self.player)
    
    def backpropagation(self, r)->None:
        if r > 0:
            self.Rate[0] += 1
        elif r == 0:
            self.Rate[1] += 1
        else:
            self.Rate[2] += 1

        if self.parent is not None:
            self.parent.backpropagation(-r)


class MCTS:
    def __init__(self, player, state, iterations=1000):
        self.player = player
        self.state = state
        self.iterations = iterations
        self.root = Node(player=player, state=state)    
    
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