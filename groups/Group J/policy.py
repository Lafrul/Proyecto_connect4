import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy


class ClassicTrialPolicy(Policy):
    TRIALS = 100

    def __init__(self) -> None:
        self.trials_per_action = self.TRIALS
        self.rng = np.random.default_rng()

    def mount(self, timeout=None) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        player = self.current_player(board)

        available_cols = [c for c in range(7) if s[0, c] == 0]

        if len(available_cols) == 0:
            raise ValueError("No valid moves available")

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
            return 1.0

        score = 0.0

        for _ in range(self.trials_per_action):
            score += self.random_trial(trial_start, -player, player)

        return score / self.trials_per_action

    def random_trial(
        self, board: np.ndarray, current_player: int, original_player: int) -> float:
        trial_board = board.copy()
        player_to_move = current_player

        while not self.is_final(trial_board):
            free_cols = [c for c in range(7) if trial_board[0, c] == 0]
            col = int(self.rng.choice(free_cols))
            self.drop_piece(trial_board, col, player_to_move)
            player_to_move = -player_to_move

        winner = ConnectState(trial_board).get_winner()
        if winner == original_player:
            return 1.0
        if winner == 0:
            return 0.0
        return -1.0

    def drop_piece(self, board: np.ndarray, col: int, player: int) -> int:
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return -1

    def is_final(self, board: np.ndarray) -> bool:
        return ConnectState(board).get_winner() != 0 or not any(board[0] == 0)

    @staticmethod
    def current_player(board: np.ndarray) -> int:
        red_moves = int(np.sum(board == -1))
        yellow_moves = int(np.sum(board == 1))
        return -1 if red_moves == yellow_moves else 1
