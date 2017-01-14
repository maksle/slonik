import cmd
import sys
from search import Engine


class Entry(cmd.Cmd):
    intro = "Slonik by M. Grinman"
    prompt = ""
    file = None

    def __init__(self):
        super(Entry, self).__init__()
        
        self.engine = Engine()
        self.engine.info = self.uci_info
        self.engine.debug_info = self.uci_debug
        self.engine.start()
        
    def uci_info(self, *info_str):
        print("info", *info_str, file=self.stdout, flush=True)
        
    def uci_debug(self, *info_str):
        self.uci_info("string", *info_str)
        
    def do_uci(self, args):
        print("id name slonik", file=self.stdout, flush=True)
        print("id author Maksim Grinman", file=self.stdout, flush=True)
        print("uciok", file=self.stdout, flush=True)

    def do_debug(self, args):
        params = args.split()
        if len(params) > 0:
            if params[0] == 'on':
                self.engine.debug = True
            elif params[0] == 'off':
                self.engine.debug = False

    def do_is_ready(self, args):
        print("readyok", file=self.stdout, flush=True)

    def do_setoption(self, args):
        return

    def do_ucinewgame(self, args):
        self.engine.stop()
        print("readyok", file=self.stdout, flush=True)

    def do_position(self, args):
        """position [fen <fenstring> | startpos ]  moves <move1> .... <movei>"""
        params = args.split()
        if args[0] == "startpos":
            args.pop(0)
            fen = ""
        elif args[0] == "fen":
            args.pop(0)
            fen = args[0]
            args.pop(0)
        moves = [arg for arg in args if arg != "moves"]
        self.engine.stop()
        self.engine.new_game(fen=fen, uci_moves=moves)

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
        entry.cleanup()
        raise
