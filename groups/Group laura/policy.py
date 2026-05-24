import math

from connect4.connect_state import ConnectState
from .mcts import mcts_uct
import numpy as np
from connect4.policy import Policy

class HashableConnectState(ConnectState):
    def __hash__(self):
        return hash((self.board.tobytes(), self.player))
    
    def __eq__(self, other):
        return (
            isinstance(other, ConnectState)
            and np.array_equal(self.board, other.board)
            and self.player == other.player
        )

class MCTSAgent(Policy):
    num_simulations = 100
    max_depth = 42 # Esta bien que este sea el maximo aunque si el estado es > 0 no se puedan hacer 42 acciones?
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

        forced_a = self.defensive_move(root_state, player)
        if forced_a is not None:
            return forced_a

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

        mcts_analysis = mcts_uct(
                            root_state,
                            legal_actions_fn,
                            successor_fn,
                            terminal_fn,
                            reward_fn,
                            num_simulations=self.num_simulations,
                            max_depth=self.max_depth,
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
        
    def defensive_move(self, root_state: ConnectState, player: int):
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
