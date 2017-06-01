#!/usr/bin/env python3 

import cmd
import sys
import logging
import logging_config
from itertools import takewhile, dropwhile
from search import Engine
import tensorflow as tf

# flags = tf.app.flags
# FLAGS = flags.FLAGS

# flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')


log = logging.getLogger(__name__)


class Entry(cmd.Cmd):
    intro = "Slonik by M. Grinman"
    prompt = ""
    
    def __init__(self, static_evaluator):
        super(Entry, self).__init__()

        self.file = open('output.log', 'w')
        
        self.p_static_evaluator = static_evaluator
        self.engine = Engine(static_evaluator)
        self.engine.info = self.uci_info
        self.engine.debug_info = self.uci_debug
        self.engine.start()
        
    def respond(self, *args):
        print(*args, file=self.stdout, flush=True)
        print(*args, file=self.file, flush=True)
        
    def precmd(self, line):
        print(line, file=self.file, flush=True)
        return line
        
    def uci_info(self, *info_str):
        self.respond(*info_str)
        
    def uci_debug(self, *info_str):
        self.uci_info("info string", *info_str)
        
    def do_uci(self, args):
        s = " (static)" if self.p_static_evaluator else ""
        self.respond("id name slonik" + s)
        self.respond("id author Maksim Grinman")
        self.respond("uciok")

    def do_debug(self, args):
        params = args.split()
        if len(params) > 0:
            if params[0] == 'on':
                self.engine.debug = True
            elif params[0] == 'off':
                self.engine.debug = False

    def do_isready(self, args):
        self.respond("readyok")

    def do_setoption(self, args):
        return

    def do_ucinewgame(self, args):
        self.engine.stop()
        self.respond("readyok")

    def do_position(self, args):
        """position [fen <fenstring> | startpos ]  moves <move1> .... <movei>"""
        params = args.split()
        if params[0] == "startpos":
            params.pop(0)
            fen = ""
        elif params[0] == "fen":
            params.pop(0)
            fen = takewhile(lambda x: x != "moves", params)
        params = dropwhile(lambda x: x != "moves", params)
        moves = [m for m in params if m != "moves"]
        self.engine.stop()
        self.engine.new_game(fen=' '.join(list(fen)), uci_moves=moves)

    def do_go(self, args):
        params = args.split()
        
        try: index = params.index("searchmoves")
        except: pass
        else: self.engine.uci_root_moves(params[index+1:].split())

        try: index = params.index("ponder")
        except: self.engine.ponder = False
        else: self.engine.ponder = True

        try: index = params.index("depth")
        except: self.engine.max_depth = None
        else: self.engine.max_depth = int(params[index+1])
        
        try: index = params.index("nodes")
        except: self.engine.max_nodes = None
        else: self.engine.max_nodes = params[index+1]

        movetime = False
        try: index = params.index("movetime")
        except: pass
        else:
            self.engine.movetime = float(params[index+1]) / 1000
            movetime = True
            
        try: index = params.index("infinite")
        except: self.engine.infinite = False
        else: self.engine.infinite = True

        time = False
        try: index = params.index("wtime")
        except: pass
        else:
            self.engine.time_management.wtime = float(params[index+1]) / 1000
            time = True
        try: index = params.index("btime")
        except: pass
        else:
            self.engine.time_management.btime = float(params[index+1]) / 1000
            time = True

        movestogo = False
        try: index = params.index("movestogo")
        except: pass
        else:
            self.engine.time_management.movestogo = int(params[index+1])
            movestogo = True

        if time and not movestogo:
            self.engine.time_management.movestogo = None

        if time and not movetime:
            self.engine.movetime = None
            
        self.engine.go()

    def do_stop(self, args):
        self.engine.stop()

    def do_ponderhit(self, args):
        self.engine.ponder = False

    def do_quit(self, args):
        self.cleanup()
        sys.exit()

    def cleanup(self):
        self.engine.quit()
        self.engine.join()
            
if __name__ == '__main__':
    static_evaluator = len(sys.argv) > 1 and sys.argv[1] == '--static'
    entry = Entry(static_evaluator)
    try:
        entry.cmdloop()
    except:
        log.exception("Top-level exception")
        entry.cleanup()
        raise
