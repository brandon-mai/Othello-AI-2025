import collections
import numpy as np
from numba import njit, types, typed, int16, int8, int64, uint64, void
from bitboard_utils import visualize_bitboard, make_move
import agents
from heuristics import disk_parity_heuristic_standalone
import minmax

board = (uint64(0x0000000810000000), uint64(0x0000001008000000))

# p1, p2 = board

# visualize_bitboard(*board)
# board = make_move(board, 19, 1)
# print(disk_parity_heuristic_standalone(board, 1))
# board = make_move(board, 18, 2)
# print(disk_parity_heuristic_standalone(board, 1))
# visualize_bitboard(*board)

a = agents.MinmaxAgent(5)
a.set_id(1)
a.get_move(board, None)

print(minmax._negamax.inspect_types())




