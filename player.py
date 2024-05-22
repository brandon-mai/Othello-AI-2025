from abc import ABC, abstractmethod
import pygame
from utils import CELL_SIZE

class Player(ABC):
    def __init__(self):
        self.id = -1
        
    def set_id(self, id):
        self.id = id

    @abstractmethod
    def get_move(self, game, valid_moves, events):
        pass

class HumanPlayer(Player):
    def get_move(self, game, valid_moves, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                if (row, col) in valid_moves:
                    return row, col
        return None