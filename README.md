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

## MinMax

### Heuristics
#### Heuristics presentation

1. **Disk Parity Heuristic**: The Disk Parity Heuristic evaluates the `difference in the number of disks between the player and the opponent`. It calculates the percentage difference relative to the total number of disks on the board. It provides a straightforward measure of dominance in terms of disk count.

2. **Mobility Heuristic**: The Mobility Heuristic assesses both `actual` and `potential` mobility for the player and the opponent. This heuristic aims to maximize the player's flexibility and options for future moves while restricting the opponent's options, thus maintaining strategic advantage.
    - **Actual mobility** refers to the number of legal moves available to the player.
    - **Potential mobility** refers to the number of potential moves (empty cells adjacent to the opponent's disks).

3. **Corner Control Heuristic**: The corner control heuristic evaluates the `control of corner squares` (A1, A8, H1, H8). It considers both the number of `corners occupied` and the `potential to occupy corners` (possible moves to corner positions). Controlling corners is highly advantageous in Othello as it provides stable positions that are difficult for the opponent to flip. This heuristic prioritizes securing these key positions.

4. **Stability Heuristic**: The stability heuristic evaluates the number of stable and unstable disks. This heuristic aims to `maximize the number of stable disks` while `minimizing the number of unstable disks`, ensuring a lasting advantage on the board.
    - **Stable disks** are those that cannot be flipped for the rest of the game.
    - **Unstable disks** are those that can be flipped at the next move of the opponent.

5. **Hybrid Heuristic**: The Hybrid Heuristic combines `multiple heuristics`, including Disk Parity, Mobility, Corner Control, and Stability, with `weighted scores`. It aims to overcome the limitations of individual heuristics by integrating multiple factors for a more comprehensive evaluation of the board state.

6. **Static Weights Heuristic**: The Static Weights Heuristic evaluates the board state by performing matrix multiplication between the board state matrix and a `predefined weight matrix`. It sums up the products to obtain the final evaluation score. This heuristic uses a predefined strategy to prioritize certain board positions over others, guiding the player towards favorable positions.

The `first five heuristics` provide a `relative evaluation` between the player and the opponent, with scores ranging `from -100 to 100`. A positive value indicates an advantage for the player, while a negative value suggests an advantage for the opponent.

#### Heuristics Analysis

To test the performances of our heuristics, we decided to make our `MinMaxAgents` play against each others for `50 games` and with `different heuristics`.


| Heuristics     | Disk Parity | Stability | Corner | Mobility | Hybrid |
|----------      |---------------|----------|----------|---------|------|
| Disk parity    | -          |  - | -       |-       | -   |
| Stability      | (50/0/0) **100%**          | -     | -        | -    | -   |
| Corner         | (40/9/1) **80%**            | (8/40/2) **16%**      | -       | -      | -   |
| Mobility       | (28/22/0) **56%**          |(0/50/0) **0%**     | (15/35/0) **30%**     | -      | -|
| Hybrid         | (50/0/0) **100%**      | (50/0/0) **100%**      | (48/2/0) **96%** | (50/0/0) **100%**    | -   |


These tests indicate the effectiveness of combining multiple evaluation criteria in the Hybrid heuristic, providing a robust strategy for Othello gameplay.
## MCTS