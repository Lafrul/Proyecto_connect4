import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

try:
    from .Euristic import HeuristicPolicyC4
except ImportError:
    import importlib

    HeuristicPolicyC4 = importlib.import_module(
        "groups.Group S.Euristic"
    ).HeuristicPolicyC4


class Winner(Policy):
    TRIALS = 50
    TOP_HEURISTIC_MOVES = 2
    SOFTMAX_TEMPERATURE = 650.0
    HEURISTIC_TIE_BREAK_WEIGHT = 0.75
    HEURISTIC_SCORE_SCALE = 5_000.0
    LEAF_SCORE_SCALE = 25_000.0

    def __init__(self) -> None:
        self.trials_per_action = self.TRIALS
        self.rng = np.random.default_rng()
        self.heuristic = HeuristicPolicyC4()

    def mount(self, timeout=None) -> None:
        self.heuristic.mount()

    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        player = self.current_player(board)

        available_cols = self.get_valid_moves(board)

        if len(available_cols) == 0:
            raise ValueError("No valid moves available")

        if np.count_nonzero(board) == 0:
            return ConnectState.COLS // 2

        scores = {
            col: self.evaluate_action(board, player, col) for col in available_cols
        }

        best_score = max(scores.values())
        best_cols = [col for col, score in scores.items() if score == best_score]

        return int(self.rng.choice(best_cols))

    def evaluate_action(self, board: np.ndarray, player: int, col: int) -> float:
        trial_start = board.copy()
        row = self.drop_piece(trial_start, col, player)

        if row < 0:
            return -np.inf

        if ConnectState(trial_start, -player).get_winner() == player:
            return 1_000_000.0

        score = 0.0

        for _ in range(self.trials_per_action):
            score += self.heuristic_trial(trial_start, -player, player)

        rollout_score = score / self.trials_per_action
        heuristic_score = self.normalized_heuristic_score(board, col, player)
        return rollout_score + self.HEURISTIC_TIE_BREAK_WEIGHT * heuristic_score

    def heuristic_trial(
        self, board: np.ndarray, current_player: int, original_player: int) -> float:
        trial_board = board.copy()
        player_to_move = current_player

        while not self.is_final(trial_board):
            col = self.select_heuristic_action(trial_board, player_to_move)
            self.drop_piece(trial_board, col, player_to_move)
            player_to_move = -player_to_move

        winner = ConnectState(trial_board).get_winner()
        if winner == original_player:
            return 1.0
        if winner == 0:
            return self.evaluate_leaf(trial_board, original_player)
        return -1.0

    def select_heuristic_action(self, board: np.ndarray, player: int) -> int:
        best_action, scores = self.heuristic.get_action(
            board, player, return_scores=True
        )
        valid_moves = self.get_valid_moves(board)
        best_score = max(scores[col] for col in valid_moves)

        if best_score >= self.heuristic.WIN_SCORE:
            return int(best_action)

        ranked_moves = sorted(valid_moves, key=lambda col: scores[col], reverse=True)
        candidates = ranked_moves[: self.TOP_HEURISTIC_MOVES]
        weights = self.heuristic_weights(scores, candidates)

        return int(self.rng.choice(candidates, p=weights))

    def heuristic_weights(
        self, scores: np.ndarray, candidates: list[int]
    ) -> np.ndarray:
        candidate_scores = np.array([scores[col] for col in candidates], dtype=float)
        candidate_scores -= float(np.max(candidate_scores))
        weights = np.exp(
            np.clip(candidate_scores / self.SOFTMAX_TEMPERATURE, -50.0, 0.0)
        )
        total_weight = float(np.sum(weights))

        if total_weight == 0.0:
            return np.full(len(candidates), 1.0 / len(candidates))

        return weights / total_weight

    def normalized_heuristic_score(
        self, board: np.ndarray, col: int, player: int
    ) -> float:
        opponent_wins_now = set(
            self.heuristic._get_winning_moves_inplace(board, -player)
        )
        raw_score = self.heuristic._evaluate_move(
            board, col, player, opponent_wins_now
        )
        if not np.isfinite(raw_score):
            return -1.0
        return float(np.tanh(raw_score / self.HEURISTIC_SCORE_SCALE))

    def evaluate_leaf(self, board: np.ndarray, player: int) -> float:
        valid_moves = self.get_valid_moves(board)
        if len(valid_moves) == 0:
            return 0.0

        _, player_scores = self.heuristic.get_action(board, player, return_scores=True)
        _, opponent_scores = self.heuristic.get_action(board, -player, return_scores=True)
        player_best = max(player_scores[col] for col in valid_moves)
        opponent_best = max(opponent_scores[col] for col in valid_moves)

        return float(np.tanh((player_best - opponent_best) / self.LEAF_SCORE_SCALE))

    def drop_piece(self, board: np.ndarray, col: int, player: int) -> int:
        for row in range(ConnectState.ROWS - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return -1

    def is_final(self, board: np.ndarray) -> bool:
        return ConnectState(board).get_winner() != 0 or not any(board[0] == 0)

    def get_valid_moves(self, board: np.ndarray) -> list[int]:
        return [col for col in range(ConnectState.COLS) if board[0, col] == 0]

    @staticmethod
    def current_player(board: np.ndarray) -> int:
        red_moves = int(np.sum(board == -1))
        yellow_moves = int(np.sum(board == 1))
        return -1 if red_moves == yellow_moves else 1
