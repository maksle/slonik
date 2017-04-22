from bb import *
import random

def get_rand_few_bits():
    return random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)

def find_magic(sq, desired_shift_offset, mask, piece_type):
    attack_fn = bishop_attack_calc if piece_type == PieceType.B else rook_attack_calc
    bits = count_bits(mask)
    desired_shift = 64 - desired_shift_offset
    occupation = []
    attacks = []
    for index in range(1 << bits):
        occ = index_to_occupation(index, bits, mask)
        free = FULL_BOARD ^ occ
        att = attack_fn(1 << sq, free)
        occupation.append(occ)
        attacks.append(att)
    for k in range(0, 100000000):
        magic = get_rand_few_bits()
        used = [None] * (1 << 12)
        fail = False
        for i in range(1 << bits):
            j = ((occupation[i] * magic) & FULL_BOARD) >> desired_shift
            if used[j] is None:
                used[j] = attacks[i]
            elif used[j] != attacks[i]:
                fail = True
                break
        if not fail:
            return magic
    return None

def find_all_magics():
    for pt in [PieceType.B, PieceType.R]:
        print("Bishop" if pt == PieceType.B else "Rook")
        for sq in range(64):
            mask = MAGIC_MASKS[pt][sq]
            print("Bishop" if pt == PieceType.B else "Rook", sq, find_magic(sq, count_bits(mask), mask, pt))

import time
now = time.time()
find_all_magics()
print(now - time.time())
