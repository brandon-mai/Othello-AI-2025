# Othello Engine with AI Agents

The primary objective of this project is to develop an Othello engine that supports different types of AI agents. This allows for the development and comparison of various AI strategies in the game of Othello.

## The Game

Othello, also known as Reversi, is a strategy board game for two players. The game is played on an `8x8 grid`, and players take turns placing discs on the board. Each player's goal is to have the `majority of their color discs on the board at the end of the game. Players can capture the opponent's discs by trapping them between two of their own, flipping the captured discs to their color.

## Project Structure

The project consists of several key components:

- `othello.py`: Contains the core game logic and the `Othello` class.
- `othello_gui.py`: Provides a GUI for playing the game using Pygame.
- `othello_simulation.py`: Enables simulation of multiple Othello games without a GUI for strategy analysis.
- `agents`: Contains different types of agents, such as `MCTSAgent`, `MinimaxAgent`, and `HumanPlayer`, which define how players make moves.
- `utils`: Contains utility functions, such as `array_utils.py` for handling board operations.
- `heuristics`: Contains heuristics functions for handling board evaluation.

## Requirements

- Python 3.7 or later
- Pygame
- Numpy
- Tqdm

You can install the required libraries using the following command:
```bash
pip install pygame numpy numba tqdm
```