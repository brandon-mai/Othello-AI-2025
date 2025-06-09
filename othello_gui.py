from os import environ
import sys

from constants import BLACK, CELL_SCALLING, CELL_SIZE, DARK_GREEN, RED, SCREEN_HEIGHT, SCREEN_WIDTH
from othello import Othello
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
from pygame import gfxdraw
from numba import uint64

from heuristics import HEURISTICS
from agents import Player, HumanPlayer, MinmaxAgent, RandomAgent, MCTSAgent

class OthelloGui:
    """
    A class to handle the Othello game using Pygame.

    Attributes:
    screen (pygame.Surface): The game screen.
    clock (pygame.time.Clock): The game clock.
    font (pygame.font.Font): The font used for rendering text.
    black_piece_img (pygame.Surface): The image for a black piece.
    white_piece_img (pygame.Surface): The image for a white piece.
    valid_black_piece_img (pygame.Surface): The image for a valid black move.
    valid_white_piece_img (pygame.Surface): The image for a valid white move.
    """
    def __init__(self, player1: Player, player2: Player):
        """
        Initializes the OthelloGUI with two players.

        Parameters:
        player1 (Player): The first player.
        player2 (Player): The second player.
        """

        self.game = Othello(player1, player2)
        self.last_move = -1
        self.init_gui()

    def init_gui(self):
        """
        Initializes pygame and all the necessary ressources for the gui.
        """
        pygame.init()
            
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Othello")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 40)
        
        # Load and scale images
        raw_black_piece_img = pygame.image.load("ressources/black_piece.png")
        raw_white_piece_img = pygame.image.load("ressources/white_piece.png")
        
        self.black_piece_img = pygame.transform.smoothscale(raw_black_piece_img, (CELL_SIZE*CELL_SCALLING, CELL_SIZE*CELL_SCALLING))
        self.white_piece_img = pygame.transform.smoothscale(raw_white_piece_img, (CELL_SIZE*CELL_SCALLING, CELL_SIZE*CELL_SCALLING))
        
        self.valid_black_piece_img = pygame.transform.smoothscale(raw_black_piece_img, (CELL_SIZE*CELL_SCALLING/2, CELL_SIZE*CELL_SCALLING/2))
        self.valid_white_piece_img = pygame.transform.smoothscale(raw_white_piece_img, (CELL_SIZE*CELL_SCALLING/2, CELL_SIZE*CELL_SCALLING/2))
        
    def draw_circle(self, color, coords, radius):
        """
        Draws an anti-aliased circle on the given surface.

        Parameters:
        surface (pygame.Surface): The surface to draw on.
        color (tuple): The color of the circle.
        coords (tuple): The (x, y) coordinates of the circle's center.
        radius (int): The radius of the circle.
        """
        x, y = coords
        gfxdraw.aacircle(self.screen, x, y, radius, color)
        gfxdraw.filled_circle(self.screen, x, y, radius, color)
    
    def draw_board(self):
        """
        Draws the game board, grid lines, pieces, and valid move indicators.
        """
        
        # Fill the screen with green
        self.screen.fill(DARK_GREEN)  
        
         # Vertical lines
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x-1, 0), (x-1, SCREEN_HEIGHT), 3) 
            
        # Horizontal lines
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y-1), (SCREEN_WIDTH, y-1), 3) 
            
        # Draw 4 dots
        for dot_x in [2, 6]:
            for dot_y in [2, 6]:
                self.draw_circle(BLACK, (dot_x * CELL_SIZE, dot_y * CELL_SIZE), 7)
                
        offset_p_pct = (1-CELL_SCALLING)/2
        
        # Draw pieces                 
        player1_pieces, player2_pieces = self.game.board
        for i in range(64):
            if (uint64(player1_pieces) >> uint64(i)) & uint64(1):
                row, col = divmod(i, 8)
                self.screen.blit(self.black_piece_img, (col * CELL_SIZE + offset_p_pct*CELL_SIZE , row * CELL_SIZE + offset_p_pct*CELL_SIZE))
            elif (uint64(player2_pieces) >> uint64(i)) & uint64(1):
                row, col = divmod(i, 8)
                self.screen.blit(self.white_piece_img, (col * CELL_SIZE + offset_p_pct*CELL_SIZE , row * CELL_SIZE + offset_p_pct*CELL_SIZE))
        
        offset_v_pct = (1-CELL_SCALLING/2)/2
                    
        # Display valid moves
        for move in self.game.current_player_moves:
            move = int(move) # numpy int8 causes overflow and whatnot in scalar multiplication
            row, col = divmod(move, 8)
            if self.game.current_player.id == 1:
                self.screen.blit(self.valid_black_piece_img, (col * CELL_SIZE + offset_v_pct*CELL_SIZE , row * CELL_SIZE + offset_v_pct*CELL_SIZE))
            else:
                self.screen.blit(self.valid_white_piece_img, (col * CELL_SIZE + offset_v_pct*CELL_SIZE , row * CELL_SIZE + offset_v_pct*CELL_SIZE))
                
        # Display last move
        row, col = divmod(self.last_move, 8)
        self.draw_circle(RED, (int((col+(1/2)) * CELL_SIZE), int((row+(1/2)) * CELL_SIZE)), 7)
        
    def display_winner(self):
        """
        Displays the winner of the game on the screen.
        """
        
        # Darken the background
        darken_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        darken_surface.set_alpha(180)  # Set transparency level to make the screen darker
        darken_surface.fill((0, 0, 0))
        self.screen.blit(darken_surface, (0, 0))

        winner = self.game.get_winner()
        if winner == 1:
            winner_text = 'Black is the winner!'
        elif winner == 2:
            winner_text = 'White is the winner!'
        else:
            winner_text = 'Draw !'
        font = pygame.font.SysFont(None, 50)
        display_text = font.render(winner_text, True, (255, 0, 0))
        text_rect = display_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(display_text, text_rect)
        
    def change_caption(self):
        """
        Changes the window caption to indicate whose turn it is.
        """
        player = "Black" if self.game.current_player.id == 1 else "White"
        text = f"Othello -- {player} Turn"
        pygame.display.set_caption(text)

    def run_game(self):
        """
        Runs the game loop with a gui, handling events, drawing the board, and managing the game state.
        """
        self.draw_board()
        pygame.display.flip()
        
        running = True
        game_over = False

        while running:
            
            events = pygame.event.get()
            
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
            
            if not game_over:
                current_player_valid_moves = self.game.current_player_moves
                if current_player_valid_moves.shape[0] == 0:
                    self.game.switch_player()
                    
                    opponent_valid_moves = self.game.current_player_moves
                    if opponent_valid_moves.shape[0] == 0:
                        game_over = True
                    continue
                  
                move = self.game.current_player.get_move(self.game.board, events)
                if move in current_player_valid_moves:
                    self.last_move = move
                    self.game.make_move(move)
            else:
                for event in events:
                    if event.type == pygame.MOUSEBUTTONDOWN: 
                        game_over = False
                        running = False
                        
            self.draw_board()
            if game_over: 
                self.display_winner()
            else: 
                self.change_caption()
            pygame.display.flip()
            self.clock.tick(30)
        
        # Solve a bug where 'pygame.quit()' fails to end the script
        pygame.display.quit() 
        
        pygame.quit()
        sys.exit()
        
if __name__ == "__main__":
    minmax_player = MinmaxAgent(depth=7, heuristic=HEURISTICS.DISK_PARITY, verbose=True)
    mcts_player = MCTSAgent(nb_iterations=250000, nb_rollouts=1, c_param=1.4, verbose=True)
    human_player = HumanPlayer()

    gui = OthelloGui(player1=human_player, 
                     player2=minmax_player)
    
    gui.run_game()
    
    
    
    
    
    