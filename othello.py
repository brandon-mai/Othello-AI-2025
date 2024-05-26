import multiprocessing

from os import environ

from tqdm import tqdm
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

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

class Othello:
    """
    A class to handle the Othello game using Pygame.

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

        self.board = np.zeros((8, 8), dtype=np.int16)
        self.board[3:5, 3:5] = [[2, 1], [1, 2]]  # Initial pieces
        
        # Create the players
        self.player1, self.player2 = player1, player2
        
        if self.player1.id == self.player2.id:
            raise Exception('Players must have different IDs')
        
        self.current_player = self.player1

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
                if self.board[row, col] == PLAYER_1:
                    self.screen.blit(self.black_piece_img, (col * CELL_SIZE + offset_p_pct*CELL_SIZE , row * CELL_SIZE + offset_p_pct*CELL_SIZE))
                elif self.board[row, col] == PLAYER_2:
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
            

    def run_game_gui(self):
        """
        Runs the game loop with a gui, handling events, drawing the board, and managing the game state.
        """
        self.init_gui()
        self.draw_board()
        
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
        
    def run_simulation(self, num_simulations):
        """
        Runs the simulation for the specified number of games without GUI.

        Returns:
        tuple: A tuple containing the number of wins for player 1, player 2, and draws.
        """
        
        if isinstance(self.player1, HumanPlayer) or isinstance(self.player2, HumanPlayer):
            raise ValueError("Cannot simulate games with HumanPlayer instances.")
        
        nb_cores = max(0, multiprocessing.cpu_count() - 1)
        
        print("=============== Othello Simulation ===============")
        print(f"Starting simulation on {nb_cores} cores.\n")
        
        with multiprocessing.Pool(processes=nb_cores) as pool:
            game_results = list(tqdm(pool.imap_unordered(self.simulate_game, ((self.player1.copy(), self.player2.copy()) for _ in range(num_simulations))), total=num_simulations))
        
        counts = {1: 0, 2: 0, 0: 0}
        for result in game_results:
            counts[result] += 1
        
        print("\n===================== Results ====================")
        print(f"Player 1 | Wins: {counts[1]:<3}, Draws: {counts[0]:<3}")
        print(f"Player 2 | Wins: {counts[2]:<3}, Draws: {counts[0]:<3}")
        
        return counts 
        
    @staticmethod
    def simulate_game(players):
        """
        Simulates a single game without GUI and returns the winner.

        Parameters:
        tuple(Player): A Tuple with both players

        Returns:
        int: 1 if player 1 wins, 2 if player 2 wins, 0 if it's a draw.
        """
        player1, player2 = players
        
        board = np.zeros((8, 8), dtype=np.int16)
        board[3:5, 3:5] = [[2, 1], [1, 2]]
        current_player = player1
        game_over = False

        while not game_over:
            opponent_id = 3 - current_player.id
            current_player_valid_moves = get_possible_moves(current_player.id, board)
            opponent_valid_moves = get_possible_moves(opponent_id, board)

            if current_player_valid_moves.size == 0 and opponent_valid_moves.size == 0:
                game_over = True
                break
            elif current_player_valid_moves.size == 0:
                current_player = player2 if current_player == player1 else player1
                continue

            move = current_player.get_move(board, None, None)
            if move in current_player_valid_moves:
                flip_tiles(move, current_player.id, board)
                current_player = player2 if current_player == player1 else player1

        count_1 = np.count_nonzero(board == 1)
        count_2 = np.count_nonzero(board == 2)
        if count_1 > count_2:
            return 1
        elif count_1 < count_2:
            return 2
        else:
            return 0
        
if __name__ == "__main__":
    game = Othello(player1=MinimaxPlayer(id=PLAYER_1, depth=5, time_limit=2, verbose=False, heuristic='hybrid'),
                   player2=MinimaxPlayer(id=PLAYER_2, depth=5, time_limit=2, verbose=False, heuristic='stability'))
    
    game.run_simulation(100)