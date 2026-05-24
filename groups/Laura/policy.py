import math

from connect4.connect_state import ConnectState
import numpy as np
from connect4.policy import Policy
import math
from typing import Any, Callable, Dict, Iterable

class HashableConnectState(ConnectState):
    def __hash__(self):
        return hash((self.board.tobytes(), self.player))
    
    def __eq__(self, other):
        return (
            isinstance(other, ConnectState)
            and np.array_equal(self.board, other.board)
            and self.player == other.player
        )

# ================ Agent ==================
class MCTSAgent(Policy):
    num_simulations = 100
    exploration_c = math.sqrt(2)
    
    def __init__(self):
        self.rng = np.random.RandomState(seed=42)

    def mount(self, timeout=None) -> None:
        self.rng = np.random.RandomState(seed=42) 

    def act(self, s: np.ndarray) -> int:
        player = self.current_player(s)
        root_state = HashableConnectState(s, player)

        if root_state.is_final():
            cols = root_state.get_free_cols()
            if cols:
                return cols[0]
            else: 
                return 0

        forced_a = self.forced_move(root_state, player)
        if forced_a is not None:
            return forced_a
        
        double_a = self.double_threat(root_state, player)
        if double_a is not None:
            return double_a

        def legal_actions_fn(state: ConnectState):
            return state.get_free_cols()
        
        def successor_fn(state: ConnectState, action: int, rng: np.random.RandomState):
            next_s = state.transition(action)
            return HashableConnectState(next_s.board, next_s.player)

        def terminal_fn(state: ConnectState):
            return state.is_final()

        def reward_fn(state: ConnectState):
            winner = state.get_winner()
            if winner == player:
                return 1
            elif winner == 0:
                return 0
            else:
                return -1
            
        fichas_jugadas = int(np.sum(s != 0))
        max_depth = 42 - fichas_jugadas

        mcts_analysis = mcts_uct(
                            root_state,
                            legal_actions_fn,
                            successor_fn,
                            terminal_fn,
                            reward_fn,
                            num_simulations=self.num_simulations,
                            max_depth=max_depth,
                            exploration_c=self.exploration_c,
                            rng=self.rng)

        best_a = mcts_analysis["best_action"]
        return best_a
    
    def current_player(self, s: np.array):
        red = 0
        yellow = 0

        for c in range(s.shape[1]):
            for r in range(s.shape[0]):
                if s[r][c] == -1:
                    red += 1
                elif s[r][c] == 1:
                    yellow += 1

        if red == yellow:
            return -1
        else:
            return 1
    
    # Si existe la posibilidad de ganar la toma y si va a perder bloquea
    def forced_move(self, root_state: ConnectState, player: int):
        cols = root_state.get_free_cols()
        
        for col in cols:
            next_s = root_state.transition(col)
            if next_s.get_winner() == player:
                return col
            
        for col in cols:
            opponent_state = HashableConnectState(root_state.board, -player)
            next_s = opponent_state.transition(col)
            if next_s.get_winner() == -player:
                return col
        
        return None
    
    # Fuerza estados donde se pueda ganar con dos posbiles acciones y 
    # evita estados en donde el oponente pueda ganar con dos posibles acciones 
    def double_threat(self, root_state: ConnectState, player: int):
        cols = root_state.get_free_cols()
        opponent = -player

        for col in cols:
            next_s = root_state.transition(col)
            future_wins = sum(1 for c in next_s.get_free_cols() if next_s.transition(c).get_winner() == player)
            if future_wins >= 2:
                return int(col)
            
        for col in cols:
            opponent_state = HashableConnectState(root_state.board, -player)
            next_s = opponent_state.transition(col)
            future_wins_opp = sum(1 for c in next_s.get_free_cols() if next_s.transition(c).get_winner() == opponent)
            if future_wins_opp >= 2:
                return int(col)
        
        return None
    
# ================ MCTS ==================
def mcts_uct(
    root_state: Any,
    legal_actions_fn: Callable[[Any], Iterable[Any]],
    successor_fn: Callable[[Any, Any, np.random.RandomState], Any],
    terminal_fn: Callable[[Any], bool],
    reward_fn: Callable[[Any], float],
    *,
    num_simulations: int,
    max_depth: int,
    exploration_c: float,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    N_s = {}
    N_sa = {}
    Q_sa = {}

    for _ in range(num_simulations):
        s = root_state
        depth = 0
        path = []

        # Selection + Expansion
        while not terminal_fn(s) and depth < max_depth and len(legal_actions_fn(s)) != 0:
            actions = legal_actions_fn(s)
            unvisited_a = [a for a in actions if (s,a) not in N_sa]

            # Si todavía hay acciones sin visitar se visitan
            if unvisited_a:
                a = unvisited_a[0]
                path.append((s,a))

                N_s[s] = N_s.get(s, 0) + 1
                N_sa[(s,a)] = 1
                Q_sa[(s,a)] = 0

                s = successor_fn(s, a, rng)
                depth += 1
                break   # para que solo expanda hasta añadir UN SOLO nuevo nodo al arbol

            # Si ya se han visitado todas las acciones se ecoge una con UCT (explore, exploit)
            else:
                best_a = None
                best_value = -float("inf")

                for a in actions:
                    uct = Q_sa[(s,a)] + exploration_c * math.sqrt(math.log(N_s.get(s,0) + 1) / (N_sa[(s,a)] + 1))
                    if uct > best_value:
                        best_value = uct
                        best_a = a
                
                a = best_a
                path.append((s,a))

                N_s[s] = N_s.get(s,0) + 1
                N_sa[(s,a)] += 1

                s = successor_fn(s, a, rng)
                depth += 1

        # Rollout inteligente
        while not terminal_fn(s) and depth < max_depth and len(legal_actions_fn(s)) != 0:
            actions = legal_actions_fn(s)
            current_player = s.player

            act = int(rng.choice(actions))
                
            opp = HashableConnectState(s.board, -current_player)
            for a in actions:
                if opp.transition(a).get_winner() == -current_player:
                    act = a
            
            for a in actions: 
                if s.transition(a).get_winner() == current_player:
                    act = a

            s = successor_fn(s, act, rng)
            depth += 1

        # Rewards
        if terminal_fn(s):
            R = reward_fn(s)
        else:
            R = 0
        
        # Backpropagation: solo guarda/actualiza los valores de los nodos que 
        # se visitan en la parte de selection + expansion
        for (state, action) in path:
            Q_sa[(state, action)] = Q_sa[(state, action)] + (R - Q_sa[(state, action)]) / N_sa[(state, action)]

    # Returns
    actions = legal_actions_fn(root_state)

    q_root = {}
    n_root = {}

    for a in actions:
        if (root_state, a) in N_sa:
            q_root[a] = Q_sa[(root_state, a)]
            n_root[a] = N_sa[(root_state, a)]

    best_action = None

    if q_root:
        max_q = max(q_root.values())
        best_candidates = [a for a in q_root if q_root[a] == max_q]
        best_action = sorted(best_candidates)[0]    

    return {"q_root": q_root, "n_root": n_root, "best_action": best_action,}