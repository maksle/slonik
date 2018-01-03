from evals import arbiter_draw, fifty_move_draw, three_fold_repetition


def has_moves(pos):
    moves = pos.generate_moves_all(legal=True)
    return len(list(moves)) > 0

def game_over(pos):
    return not has_moves(pos) or arbiter_draw(pos)


class Node(object):
    def __init__(self):
        self.N = 0


class Edge(object):
    def __init__(self):
        self.N = 0
        self.Q = 0


class MCTS(object):
    def __init__(self, s0, max_simulations=800, c=math.sqrt(2), w_r=.5):
        self.max_simulations = max_simulations
        self.simulations = 0

        # Root state to search from
        self.s0 = s0

        # Exploration parameter. Is 0.05 in ExIT.
        # Not used in AlphaGoZero, presumably same in AlphaZero.
        # Theoretical value is sqrt(2).
        # ExIT use c = 0.05 but they use RAVE which affects exploration.
        self.c = c 

        # c_RAVE is 3000 in ExIT.
        # AlphaZero paper thinks RAVE is unnecessary when using Policy Networks.
        # ExIT paper thinks RAVE helps efficiency early on.

        # Mixing parameter for rollout vs NN value estimate
        self.w_r = wr 

        # UCT bonus towards action prior. AlphaZero essentially uses this as
        # their main UCT term with a slightly different formula, so this is
        # essentially the exploration parameter there.
        # AlphaGoZero seemed to use w_a = 5 * additional term sqrt(node.N).
        # ExIT paper recommends w_a = avg simulations per action in the root
        # e.g. 100 in game of Go when doing 10k simulations.
        self.w_a = self.max_simulations / 35

        # Value bonus for UCT. Not used in Alpha*. ExIT used 0.75.
        self.w_v = 0.75

        self.tree = dict()
    
    def search(self):
        pos0 = Position.from_fen(self.s0)
        while self.time_available():
            self.simulate(Position(pos0))
        return select_move(pos0, 0)

    def time_available(self):
        return self.simulations < self.max_simulations
    
    def simulate(self, position):
        self.simulations += 1
        states_actions = self.sim_tree(position)
        z = self.sim_default(Position(position)) if self.w_r > 0 else 0
        self.backup(states_actions, z)
        
    def sim_tree(self, position):
        states_actions = []
        while not game_over(position):
            s = get_state(position)
            if s not in tree:
                self.new_node(s)
                states_actions.append((s, None))
                break
            a = self.select_move(position, self.c)
            states_actions.append((s, a.to_uci))
            position.make_move(a)
        return states_actions

    def default_policy(self, position):
        legal = self.get_actions(position=position)
        return np.random.choice(legal)
    
    def sim_default(self, position):
        while not game_over(position):
            a = self.default_policy(position)
            position.make_move(a)
        stm = position.side_to_move()
        if stm == Side.B and position.in_check(): return 1.0
        elif stm == Side.W and position.in_check(): return -1.0
        else: return 0.0

    def new_node(self, s):
        node = Node()
        node.N = 0
        self.tree[s] = node
        for a in self.get_actions(s=s):
            edge = Edge()
            # edge.Pnn = ...
            # edge.Qnn = ...
            edge.N = 0
            edge.Q = 0
            self.tree[(s, a.to_uci)] = edge
    
    def uct_value(self, s, a, c, w_a, w_v):
        node = self.tree[s]
        edge = self.tree[(s, a.to_uci)]
        
        uct = c * math.sqrt(math.log(node.N) / edge.N)

        # May want to tune temperature of the softmax generating P towards
        # optimizing this bonus using Pnn. ExIT use temp 1.0. AlphaGoZero used
        # 0.67 for temp.
        policy_prior_bonus = w_a * edge.Pnn / (edge.N + 1)
        
        value_prior_bonus = w_v * edge.Qnn

        return uct + policy_prior_bonus + value_prior_bonus

    def Q(self, s, a):
        edge = self.tree[(s, a.to_uci)]
        return edge.Q
    
    def select_move(self, position, c):
        # When c == 0 should we select just based on largest N(s,a)?
        # Perhaps extend the search if argmax Q(s,a) and argmax N(s,a) disagree
        s = self.get_state(position)
        legal = self.get_actions(position=position)
        uct = self.uct_value(s, a, c, self.w_a, self.w_v)
        if position.white_to_move():
            uct_vals = [self.Q(s, a) + uct for a in legal]
            best_ind = np.argmax(uct_vals)
        else:
            uct_vals = [self.Q(s, a) - uct for a in legal]
            best_ind = np.argmin(uct_vals)
        return legal[best_ind]

    def backup(self, states_actions, z):
        for s, a in state_actions:
            node = self.tree[s]
            node.N += 1
            if a is not None:
                edge = self.tree[(s, a.to_uci)]
                edge.N += 1
                v = (1 - self.w_r) * edge.Qnn + self.w_r * z
                edge.Q += (v - edge.Q) / edge.N

    def get_actions(self, s=None, position=None):
        pos = position if position else Position.from_fen(s)
        return list(pos.generate_moves_all(legal=True))

    def get_state(self, position):
        return position.fen(timeless=True)
