from IPython import embed
import chess
import chess.pgn
import random
import pandas as pd
import os
from multiprocessing import Manager, Process


output_file_name_template = '../../slonik_data/positions_{}.pkl'

def do_work(process_num, in_queue):
    file_name = output_file_name_template.format(process_num)
    if os.path.exists(file_name):
        data = pd.read_pickle(file_name)
    else:
        data = pd.DataFrame(columns=('fen', 'result'))

    n = 0
    while True:
        game = in_queue.get()

        if game == None:
            data.to_pickle(file_name)
            return
        
        if len(game.errors):
            continue
        moves = list(game.main_line())
        if len(moves) < 10:
            continue

        stop = random.randrange(0, len(moves))
        b = game.board()
        for move in moves[:stop]:
            b.push(move)
        
        game_result = game.headers['Result']
        if game_result == '1-0': game_result = '1'
        elif game_result == '0-1': game_result = '-1'
        elif game_result == '1/2-1/2': game_result = '0'
        else: continue
            
        data.loc[data.shape[0]] = [b.fen().strip(), game_result]

        n += 1
        if n % 999 == 0:
            print('saving', n)
            data.to_pickle(file_name)

if __name__ == '__main__':
    pgn_file_name = '/home/maks/scid/allgames.pgn'

    num_workers = os.cpu_count()
    manager = Manager()
    work = manager.Queue(num_workers)

    pool = []
    for i in range(num_workers):
        p = Process(target=do_work, args=(i, work))
        p.start()
        pool.append(p)

    with open(pgn_file_name, 'r') as pgn:
        n = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                for i in range(num_workers): work.put(None)
                break
            else:
                # print(game)
                work.put(game)
            n += 1
                
    for p in pool:
        p.join()
    
# data.to_pickle(output_file_name)
# print("Size:", data.size)
print("Done")
