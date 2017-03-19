#!/usr/bin/env python3 

import cmd
import sys
import logging
import logging_config
from itertools import takewhile, dropwhile
from search import Engine

log = logging.getLogger(__name__)


class Entry(cmd.Cmd):
    intro = "Slonik by M. Grinman"
    prompt = ""
    
    def __init__(self):
        super(Entry, self).__init__()

        self.file = open('output.log', 'w')
        
        self.engine = Engine()
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
        self.respond("id name slonik")
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
        else: self.engine.init_root_moves(params[index+1:].split())

        try: index = params.index("ponder")
        except: pass
        else: self.engine.ponder = True

        try: index = params.index("depth")
        except: pass
        else: self.engine.max_depth = params[index+1]
        
        try: index = params.index("nodes")
        except: pass
        else: self.engine.max_nodes = params[index+1]

        try: index = params.index("movetime")
        except: pass
        else: self.engine.movetime = float(params[index+1]) / 1000
            
        try: index = params.index("infinite")
        except: pass
        else: self.engine.infinite = True
            
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
    entry = Entry()
    try:
        entry.cmdloop()
    except:
        log.exception("Top-level exception")
        entry.cleanup()
        raise
