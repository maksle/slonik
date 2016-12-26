import random
import math
from enum import Enum

class TTEntry():
    def __init__(self, key=0, move=0, bound_type=0, value=0, depth=0, static_eval=None):
        self.key = key
        self.move = move
        self.bound_type = bound_type
        self.value = value
        self.depth = depth
        self.static_eval = static_eval

class BoundType(Enum):
    EXACT = 1
    LO_BOUND = 2
    HI_BOUND = 3
    NONE = 4

def make_tt():
    tt = [None] * (1<<25)
    return tt, len(tt)

def get_rand64():
    return random.getrandbits(64)

def get_rand_array(n):
    arr = []
    for i in range(0,n):
        arr.append(get_rand64())
    return arr

def get_tt_index(key):
    index = key % TT_SIZE
    # if TT[index] is not None and TT[index].key != key:
    #     print('collission')
    if TT[index] is not None and TT[index].key == key:
        return True, index, TT[index]
    else:
        return False, index, TT[index]

def save_tt_entry(tt_entry):
    found, index, found_entry = get_tt_index(tt_entry.key)
    TT[index] = tt_entry

print("Initializing zobrists")
ZOBRIST_PIECE_SQUARES = []
for i in range(0,64):
    piece_arr = []
    ZOBRIST_PIECE_SQUARES.append(piece_arr)
    piece_arr.append(0)
    for j in range(1,13):
        piece_arr.append(random.getrandbits(64))
ZOBRIST_SIDE = get_rand_array(2)
ZOBRIST_CASTLE = get_rand_array(4)

TT, TT_SIZE = make_tt()
