import json
from pathlib import Path

import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy


class BaseLuisAgent(Policy):
    rows = 6
    cols = 7
    center_order = [3, 2, 4, 1, 5, 0, 6]

    def __init__(self, weight_file: str | None = None):
        self.weight_file = weight_file
        self.opponent_weight = 0.8
        self.weights = {
            "bias": 0.08803236930065596,
            "center": 2.0853886703131446,
            "own_two": -0.33705938106525335,
            "own_three": 1.9233357770046813,
            "opp_two": 0.30497077125357164,
            "opp_three": -2.070941027019447,
            "height": -0.12487660033640015,
        }

    def mount(self, timeout=None) -> None:
        if self.weight_file is None:
            return
        path = Path(__file__).with_name(self.weight_file)
        if path.exists():
            with open(path, "r") as file:
                self.weights.update(json.load(file))

    def act(self, s: np.ndarray) -> int:
        board = s.copy()
        player = self.current_player(board)
        valid = self.valid_columns(board)

        move = self.winning_move(board, player)
        if move is not None:
            return move

        move = self.winning_move(board, -player)
        if move is not None:
            return move

        scores = {}
        for col in valid:
            next_board = self.play_move(board, col, player)
            scores[col] = self.value(next_board, player)
            if self.winning_move(next_board, -player) is not None:
                scores[col] -= 500.0
            if self.opponent_weight > 0.0:
                scores[col] -= self.opponent_weight * self.opponent_answer(
                    next_board, player
                )

        best_score = max(scores.values())
        best_cols = [
            col
            for col in self.center_order
            if col in scores and scores[col] == best_score
        ]
        return best_cols[0]

    def current_player(self, board: np.ndarray) -> int:
        red = int(np.sum(board == -1))
        yellow = int(np.sum(board == 1))
        if red == yellow:
            return -1
        return 1

    def valid_columns(self, board: np.ndarray) -> list[int]:
        return [col for col in range(self.cols) if board[0, col] == 0]

    def play_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        next_board = board.copy()
        for row in range(self.rows - 1, -1, -1):
            if next_board[row, col] == 0:
                next_board[row, col] = player
                return next_board
        return next_board

    def winning_move(self, board: np.ndarray, player: int) -> int | None:
        for col in self.center_order:
            if board[0, col] != 0:
                continue
            next_board = self.play_move(board, col, player)
            if ConnectState(next_board).get_winner() == player:
                return col
        return None

    def opponent_answer(self, board: np.ndarray, player: int) -> float:
        values = []
        for col in self.valid_columns(board):
            next_board = self.play_move(board, col, -player)
            values.append(self.value(next_board, -player))
        if not values:
            return 0.0
        return max(values)

    def value(self, board: np.ndarray, player: int) -> float:
        winner = ConnectState(board).get_winner()
        if winner == player:
            return 1000.0
        if winner == -player:
            return -1000.0
        if not any(board[0] == 0):
            return 0.0

        features = self.features(board, player)
        total = 0.0
        for name, amount in features.items():
            total += self.weights.get(name, 0.0) * amount
        return total

    def features(self, board: np.ndarray, player: int) -> dict[str, float]:
        features = {
            "bias": 1.0,
            "center": 0.0,
            "own_two": 0.0,
            "own_three": 0.0,
            "opp_two": 0.0,
            "opp_three": 0.0,
            "height": 0.0,
        }

        center_col = board[:, 3]
        features["center"] = float(
            np.sum(center_col == player) - np.sum(center_col == -player)
        )

        for col in range(self.cols):
            pieces = int(np.sum(board[:, col] != 0))
            features["height"] += pieces * pieces

        windows = self.all_windows(board)
        for window in windows:
            own = window.count(player)
            opp = window.count(-player)
            empty = window.count(0)
            if opp == 0 and empty > 0:
                if own == 2:
                    features["own_two"] += 1.0
                elif own == 3:
                    features["own_three"] += 1.0
            if own == 0 and empty > 0:
                if opp == 2:
                    features["opp_two"] += 1.0
                elif opp == 3:
                    features["opp_three"] += 1.0

        return features

    def all_windows(self, board: np.ndarray) -> list[list[int]]:
        windows = []

        for row in range(self.rows):
            for col in range(self.cols - 3):
                windows.append([int(board[row, col + i]) for i in range(4)])

        for row in range(self.rows - 3):
            for col in range(self.cols):
                windows.append([int(board[row + i, col]) for i in range(4)])

        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                windows.append([int(board[row + i, col + i]) for i in range(4)])

        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                windows.append([int(board[row + i, col - i]) for i in range(4)])

        return windows


class LuisTactical(BaseLuisAgent):
    def __init__(self):
        super().__init__(None)
        self.weights = {
            "bias": 0.0,
            "center": 1.0,
            "own_two": 0.0,
            "own_three": 0.0,
            "opp_two": 0.0,
            "opp_three": 0.0,
            "height": 0.0,
        }


class LuisValueRandom(BaseLuisAgent):
    def __init__(self):
        super().__init__("weights_random.json")
        self.weights = {
            "bias": 0.06700138936735955,
            "center": 2.0761175826878353,
            "own_two": -0.3424649156918519,
            "own_three": 1.919376504062442,
            "opp_two": 0.30865202503720046,
            "opp_three": -2.065396543315938,
            "height": -0.1269181314342875,
        }
        self.opponent_weight = 0.4


class LuisValueSelfPlay(BaseLuisAgent):
    def __init__(self):
        super().__init__("weights_self_play.json")
        self.weights = {
            "bias": 0.08803236930065596,
            "center": 2.0853886703131446,
            "own_two": -0.33705938106525335,
            "own_three": 1.9233357770046813,
            "opp_two": 0.30497077125357164,
            "opp_three": -2.070941027019447,
            "height": -0.12487660033640015,
        }
        self.opponent_weight = 0.8


class Winner(LuisValueSelfPlay):
    """Default Mario policy exposed to the tournament loader."""

