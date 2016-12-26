def print_bb(bb):
    """Prints bitboard"""
    digits = bin(bb)[2:][::-1][:64][::-1]
    len_digits = len(digits)
    if len_digits < 64:
        fill = '0' * (64 - len_digits)
        digits = fill + digits
    print(digits[:8],
          digits[8:16],
          digits[16:24],
          digits[24:32],
          digits[32:40],
          digits[40:48],
          digits[48:56],
          digits[56:64], sep='\n')

