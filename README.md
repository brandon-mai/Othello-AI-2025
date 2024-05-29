# Othello Engine with AI Agents

![Simulation of an Othello Game](./doc/Othello%20--%20Black%20Turn%202024-05-29%2017-31-45.gif)

The primary objective of this project is to develop an Othello engine that supports different types of AI agents. This allows for the development and comparison of various AI strategies in the game of Othello.

## The Game

Othello, also known as Reversi, is a strategy board game for two players. The game is played on an `8x8 grid`, and players take turns placing discs on the board. Each player's goal is to have the majority of their color discs on the board at the end of the game. Players can capture the opponent's discs by trapping them between two of their own, flipping the captured discs to their color.

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
- Numba
- Numpy
- Tqdm

You can install the required libraries using the following command:
```bash
pip install pygame numpy numba tqdm
```

## How to launch

 TODO

## MiniMax

The Minimax algorithm is a recursive method used in decision-making and game theory to determine the optimal move for a player, assuming that the opponent also plays optimally. The algorithm explores all possible moves to a specified depth and evaluates the resulting board states using an evaluation function. The main idea is to maximize the player's minimum gain, which leads to the name "Minimax."


### Heuristics
When employing algorithms like Minimax, heuristics help to assess the desirability of different board states, guiding the AI in its gameplay strategy. Heuristics are rules of thumb or strategies used to make quick, efficient decisions that are not guaranteed to be perfect but are practical and sufficient for the task at hand. In the context of Othello, heuristics evaluate the board state to determine which moves are likely to lead to a favorable outcome.

#### Heuristics presentation

1. **Disk Parity Heuristic**: The Disk Parity Heuristic evaluates the `difference in the number of disks between the player and the opponent`. It calculates the percentage difference relative to the total number of disks on the board. It provides a straightforward measure of dominance in terms of disk count.

2. **Mobility Heuristic**: The Mobility Heuristic assesses both `actual` and `potential` mobility for the player and the opponent. This heuristic aims to maximize the player's flexibility and options for future moves while restricting the opponent's options, thus maintaining strategic advantage.
    - **Actual mobility** refers to the number of legal moves available to the player.
    - **Potential mobility** refers to the number of potential moves (empty cells adjacent to the opponent's disks).

3. **Corner Control Heuristic**: The corner control heuristic evaluates the `control of corner squares` (A1, A8, H1, H8). It considers both the number of `corners occupied` and the `potential to occupy corners` (possible moves to corner positions). Controlling corners is highly advantageous in Othello as it provides stable positions that are difficult for the opponent to flip. This heuristic prioritizes securing these key positions.

4. **Stability Heuristic**: The stability heuristic evaluates the number of stable and unstable disks. This heuristic aims to `maximize the number of stable disks` while `minimizing the number of unstable disks`, ensuring a lasting advantage on the board.
    - **Stable disks** are those that cannot be flipped for the rest of the game.
    - **Unstable disks** are those that can be flipped at the next move of the opponent.

5. **Hybrid Heuristic**: The Hybrid Heuristic combines `multiple heuristics`, including Disk Parity, Mobility, Corner Control, and Stability, with `weighted scores`. It aims to overcome the limitations of individual heuristics by integrating multiple factors for a more comprehensive evaluation of the board state. The current weights are the following :
   - `Disk Parity Heuristic`: $ ({1 + ({number\_of\_disks \over 64})})^6 $ This scales exponentially as the game progresses.
   - `Mobility Heuristic`: 20
   - `Corner Control Heuristic`: 50
   - `Stability Heuristic`: 40

6. **Static Weights Heuristic**: The Static Weights Heuristic evaluates the board state by performing matrix multiplication between the board state matrix and a `predefined weight matrix`. It sums up the products to obtain the final evaluation score. This heuristic uses a predefined strategy to prioritize certain board positions over others, guiding the player towards favorable positions.

The `first five heuristics` provide a `relative evaluation` between the player and the opponent, with scores ranging `from -100 to 100`. A positive value indicates an advantage for the player, while a negative value suggests an advantage for the opponent.

#### Heuristics Analysis

To test the performances of our heuristics, we decided to make our `MinMaxAgents` play against each others for `50 games` and with `different heuristics`.


| Heuristics     | Disk Parity | Stability | Corner | Mobility | Hybrid |
|----------      |---------------|----------|----------|---------|------|
| Disk parity    | -   |  - | -  |-  | -   |
| Stability      | (50/0/0) **100%**  | -   | -  | -    | -   |
| Corner         | (40/9/1) **80%**  | (8/40/2) **16%**      | -       | -      | -   |
| Mobility       | (6/44/0) **12%**  |(0/50/0) **0%** | (15/35/0) **30%** | -   | -|
| Hybrid         | (50/0/0) **100%**  | (50/0/0) **100%** | (48/2/0) **96%** | (50/0/0) **100%**  | -   |


The results revealed interesting insights into each heuristic's performance. The **Stability heuristic stood out as particularly** promising, achieving a perfect win rate against all other heuristics except for the Corner heuristic, against which it still performed well but not flawlessly. This implies that the **Corner heuristic's strategy of quickly securing corner positions can effectively disrupt Stability’s aim to establish stable disks**. Disk Parity, as expected, performed poorly, demonstrating that **having the most disks before the game's end is not necessarily an advantage**. Instead, it provides more mobility to the opponent, allowing them to capture more pieces.

The **Mobility heuristic was the least effective**, as it focuses solely on maximizing the number of legal moves without considering positional strength. **This heuristic likely works better when combined with others**, but on its own, it does not perform well.

Finally, the **Hybrid heuristic outperformed all other heuristics**, showcasing the **effectiveness of combining multiple evaluation criteria**. These results indicate that a multifaceted approach provides a robust and adaptable strategy for Othello gameplay.

### Improvements Implemented

#### 1. **Negamax Algorithm**
Negamax is a streamlined version of the Minimax algorithm, specifically designed for zero-sum games. It leverages the property \( \max(a, b) = -\min(-a, -b) \), which simplifies the recursive evaluation process. Instead of having separate maximization and minimization functions, Negamax uses a single function where the perspective of the opponent is represented by negating the score. This means that the maximizing player’s gain is equivalent to the minimizing player’s loss. By always maximizing from the current player's perspective and negating the score when switching turns, Negamax reduces the complexity of the implementation and potential errors associated with handling two different functions.

The Negamax algorithm operates as follows:
```
function negamax(node, depth, color) is
    if depth = 0 or node is a terminal node then
        return color × the heuristic value of node
    value := −∞
    for each child of node do
        value := max(value, −negamax(child, depth − 1, −color))
    return value
```

#### 2. **Alpha-Beta Pruning**
Alpha-Beta pruning enhances the Negamax algorithm by eliminating branches in the game tree that cannot influence the final decision. It introduces two values, alpha and beta, which represent the minimum score that the maximizing player is assured of and the maximum score that the minimizing player is assured of, respectively. As the algorithm explores the tree, these values are updated and used to prune branches that are not promising. If a branch is found where the score is worse than a previously examined option for the player, further exploration of that branch is stopped. This significantly reduces the number of nodes evaluated, enabling deeper searches within the same computational limits.+

#### 3. **Iterative Deepening with Time Constraint**
Iterative deepening is a search strategy where the algorithm progressively deepens the search one level at a time until a time limit is reached. This approach ensures that the best possible move is found within the given time constraint. By starting with a shallow search and gradually increasing the depth, the algorithm can return the best move found so far if the time runs out. This provides an anytime characteristic, meaning that it can be interrupted and still return a valid move. It combines the benefits of depth-first and breadth-first search, allowing the agent to adapt its search depth dynamically based on the available time.

#### 4. **Transposition Table and Zobrist Hashing**
The transposition table, coupled with Zobrist hashing, acts as a memory cache, storing previously evaluated board states to avoid redundant calculations. When the algorithm encounters a board state, it computes its Zobrist hash and checks if the state is already in the transposition table. If found, it retrieves the stored evaluation, saving precious computational resources. Zobrist hashing, by assigning unique bitstrings to each piece position and type, ensures collision-resistant hash keys for board states. This technique realy shines when integrated with iterative deepening, it enhances the efficiency of the search process by avoiding redundant evaluations across varying depths.

#### 5. **Move Ordering**
Move Ordering optimizes the search process by arranging possible moves based on a heuristic before evaluation. These scores estimate the potential strength of each move, allowing the algorithm to prioritize more promising options. By evaluating stronger moves first, Alpha-Beta pruning becomes more effective, cutting off unpromising branches early in the search. This focused exploration of the most influential moves enhances the efficiency of the search, leading to faster convergence on the optimal move. In our implementation, if a move is found in the transposition table for a given position, it is tested first. Otherwise, the remaining moves are sorted using the static weight heuristic, ensuring that the most promising moves are considered early in the search process.

#### 6. **MTD(f) Algorithm**
MTD(f) is an optimization technique used within iterative deepening to refine the search process. It stands for Memory-enhanced Test Driver with a first guess and combines elements of Negamax and Alpha-Beta pruning. The algorithm starts with an initial guess for the value of the board state and performs a series of narrow alpha-beta searches to converge on the optimal move. By refining the search boundaries in small increments, MTD(f) quickly hones in on the best move, reducing the number of full-width searches needed. This makes the iterative deepening process more efficient, leading to faster convergence on the optimal move.

#### 7. **Bitboards**
Bitboards enable rapid and streamlined manipulation of the game state through bitwise operations. This efficiency extends to various tasks such as flipping pieces, verifying legal moves, and updating the board, resulting in faster execution times compared to traditional array-based implementations. Moreover, bitboards consume less memory, making them well-suited for resource-constrained environments.

Currently, bitboards are utilized primarily within the heuristic functions to perform calculations efficiently. However, there are plans to extend their usage to other parts of the algorithm in the future. By incorporating bitboards more extensively, we aim to enhance the overall performance of the algorithm, enabling faster and more optimized decision-making processes across all components.

### Potential Improvments
- Improve the algorithm ressible for finding possible moves with bitboards.
- Change the way a move is represented  (store the starting cell, the direction(s) and the number of disks captured). This will improve the flipping process as we will no longer need to check every directions.
- Extend the usage of Bitboards to the whole project and get rid of the old Array-wise operations.
- Use Late Move Reduction (LMR) to make a more aggressive pruning and only evaluate the msot promissing moves at full depth. 




## MCTS