import pygame
from pygame import gfxdraw

from minmax import MinimaxPlayer
from player import HumanPlayer
from array_utils import *

def draw_circle(surface, color, coords, radius):
    """
    Draws an anti-aliased circle on the given surface.

    Parameters:
    surface (pygame.Surface): The surface to draw on.
    color (tuple): The color of the circle.
    coords (tuple): The (x, y) coordinates of the circle's center.
    radius (int): The radius of the circle.
    """
    x, y = coords
    gfxdraw.aacircle(surface, x, y, radius, color)
    gfxdraw.filled_circle(surface, x, y, radius, color)

class OthelloGUI:
    """
    A class to handle the Othello game GUI using Pygame.

    Attributes:
    screen (pygame.Surface): The game screen.
    clock (pygame.time.Clock): The game clock.
    font (pygame.font.Font): The font used for rendering text.
    board (np.ndarray): The game board.
    player1 (Player): The first player.
    player2 (Player): The second player.
    current_player (Player): The player whose turn it is.
    black_piece_img (pygame.Surface): The image for a black piece.
    white_piece_img (pygame.Surface): The image for a white piece.
    valid_black_piece_img (pygame.Surface): The image for a valid black move.
    valid_white_piece_img (pygame.Surface): The image for a valid white move.
    """
    def __init__(self, player1, player2):
        """
        Initializes the OthelloGUI with two players.

        Parameters:
        player1 (Player): The first player.
        player2 (Player): The second player.
        """
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Othello")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 40)
        
        self.board = np.zeros((8, 8), dtype=np.int16)
        self.board[3:5, 3:5] = [[2, 1], [1, 2]]  # Initial pieces
        
        # Create the players
        self.player1, self.player2 = player1, player2
        self.player1.set_id(PLAYER1)
        self.player2.set_id(PLAYER2)
        
        self.current_player = self.player1
        
        # Load and scale images
        raw_black_piece_img = pygame.image.load("ressources/black_piece.png")
        raw_white_piece_img = pygame.image.load("ressources/white_piece.png")
        
        self.black_piece_img = pygame.transform.smoothscale(raw_black_piece_img, (CELL_SIZE*CELL_SCALLING, CELL_SIZE*CELL_SCALLING))
        self.white_piece_img = pygame.transform.smoothscale(raw_white_piece_img, (CELL_SIZE*CELL_SCALLING, CELL_SIZE*CELL_SCALLING))
        
        self.valid_black_piece_img = pygame.transform.smoothscale(raw_black_piece_img, (CELL_SIZE*CELL_SCALLING/2, CELL_SIZE*CELL_SCALLING/2))
        self.valid_white_piece_img = pygame.transform.smoothscale(raw_white_piece_img, (CELL_SIZE*CELL_SCALLING/2, CELL_SIZE*CELL_SCALLING/2))

        
    def switch_player(self):
        """
        Switches the current player to the other player.
        """
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
    
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
                draw_circle(self.screen, BLACK, (dot_x * CELL_SIZE, dot_y * CELL_SIZE), 7)
                
        offset_p_pct = (1-CELL_SCALLING)/2
        
        # Draw pieces
        for row in range(8):
            for col in range(8):
                if self.board[row, col] == PLAYER1:
                    self.screen.blit(self.black_piece_img, (col * CELL_SIZE + offset_p_pct*CELL_SIZE , row * CELL_SIZE + offset_p_pct*CELL_SIZE))
                elif self.board[row, col] == PLAYER2:
                    self.screen.blit(self.white_piece_img, (col * CELL_SIZE + offset_p_pct*CELL_SIZE , row * CELL_SIZE + offset_p_pct*CELL_SIZE))
        
        offset_v_pct = (1-CELL_SCALLING/2)/2
                    
        # Display valid moves
        for row, col in get_possible_moves(self.current_player.id, self.board):
            if self.current_player.id == 1:
                self.screen.blit(self.valid_black_piece_img, (col * CELL_SIZE + offset_v_pct*CELL_SIZE , row * CELL_SIZE + offset_v_pct*CELL_SIZE))
            else:
                self.screen.blit(self.valid_white_piece_img, (col * CELL_SIZE + offset_v_pct*CELL_SIZE , row * CELL_SIZE + offset_v_pct*CELL_SIZE))

    def display_winner(self):
        """
        Displays the winner of the game on the screen.
        """
        # Darken the background
        darken_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        darken_surface.set_alpha(180)  # Set transparency level to make the screen darker
        darken_surface.fill((0, 0, 0))
        self.screen.blit(darken_surface, (0, 0))

        winner = self.get_winner()
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
        
    def get_winner(self):
        """
        Determines the winner of the game based on the piece counts.

        Returns:
        int: 1 if player 1 wins, 2 if player 2 wins, 0 if it's a draw.
        """
        count_1 = np.count_nonzero(self.board == 1)
        count_2 = np.count_nonzero(self.board == 2)
        if count_1 > count_2:
            return 1
        elif count_1 < count_2:
            return 2
        else:
            return 0
        
    def change_caption(self):
        """
        Changes the window caption to indicate whose turn it is.
        """
        player = "Black" if self.current_player.id == 1 else "White"
        text = f"Othello -- {player} Turn"
        pygame.display.set_caption(text)
        
    def make_move(self, row, col):
        """
        Makes a move by the current player and flips the appropriate tiles.

        Parameters:
        row (int): The row index of the move.
        col (int): The column index of the move.
        """
        flip_tiles((row, col), self.current_player.id, self.board)
        self.switch_player()
            

    def run_game(self):
        """
        Runs the main game loop, handling events, drawing the board, and managing the game state.
        """
        running = True
        game_over = False

        while running:
            
            self.change_caption()
            events = pygame.event.get()
            
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
            
            opponent = 3 - self.current_player.id
            
            current_player_valid_moves = get_possible_moves(self.current_player.id, self.board)
            opponent_valid_moves = get_possible_moves(opponent, self.board)
            
            if not game_over and (current_player_valid_moves.size == 0 and opponent_valid_moves.size == 0):
                game_over = True
            
            elif not game_over and current_player_valid_moves.size == 0:
                self.switch_player()
                continue
            
            if not game_over:       
                move = self.current_player.get_move(self.board, current_player_valid_moves, events)
                
                if move in current_player_valid_moves:
                    self.make_move(*move)
            else:
                for event in events:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        game_over = False
                        running = False
                        
            self.clock.tick(30)
            self.draw_board()
            if game_over: self.display_winner()
            pygame.display.flip()
            
        pygame.quit()
        
if __name__ == "__main__":
    pygame.init()
    game_gui = OthelloGUI(player1=MinimaxPlayer(3),
                          player2=MinimaxPlayer(3))
    game_gui.draw_board()
    game_gui.run_game()