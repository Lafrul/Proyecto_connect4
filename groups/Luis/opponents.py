import math

import numpy as np

from connect4.connect_state import ConnectState


class RandomAgent:
    def __init__(self):
        self.rng = np.random.default_rng()

    def mount(self) -> None:
        pass

    def act(self, board: np.ndarray) -> int:
        valid = [col for col in range(7) if board[0, col] == 0]
        return int(self.rng.choice(valid))


class SebastianTrialAgent:
    rows = 6
    cols = 7
    trials = 100

    def __init__(self):
        self.rng = np.random.default_rng()

    def mount(self) -> None:
        pass

    def act(self, board: np.ndarray) -> int:
        player = self.current_player(board)
        valid = [col for col in range(self.cols) if board[0, col] == 0]
        scores = {}

        for col in valid:
            scores[col] = self.score_action(board, player, col)

        best = max(scores.values())
        best_cols = [col for col, score in scores.items() if score == best]
        return int(self.rng.choice(best_cols))

    def score_action(self, board: np.ndarray, player: int, col: int) -> float:
        next_board = self.play_move(board, col, player)

        if ConnectState(next_board).get_winner() == player:
            return 1.0

        total = 0.0
        for _ in range(self.trials):
            total += self.random_game(next_board, -player, player)

        return total / self.trials

    def random_game(self, board: np.ndarray, turn: int, player: int) -> float:
        state_board = board.copy()
        current = turn

        while not self.is_final(state_board):
            valid = [col for col in range(self.cols) if state_board[0, col] == 0]
            col = int(self.rng.choice(valid))
            state_board = self.play_move(state_board, col, current)
            current = -current

        winner = ConnectState(state_board).get_winner()
        if winner == player:
            return 1.0
        if winner == 0:
            return 0.0
        return -1.0

    def play_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        next_board = board.copy()
        for row in range(self.rows - 1, -1, -1):
            if next_board[row, col] == 0:
                next_board[row, col] = player
                return next_board
        return next_board

    def is_final(self, board: np.ndarray) -> bool:
        return ConnectState(board).get_winner() != 0 or not any(board[0] == 0)

    def current_player(self, board: np.ndarray) -> int:
        red = int(np.sum(board == -1))
        yellow = int(np.sum(board == 1))
        if red == yellow:
            return -1
        return 1


class LauraMCTSAgent:
    rows = 6
    cols = 7
    simulations = 100
    max_depth = 42
    exploration = math.sqrt(2)
    center_order = [3, 2, 4, 1, 5, 0, 6]

    def __init__(self):
        self.rng = np.random.default_rng(42)

    def mount(self) -> None:
        self.rng = np.random.default_rng(42)

    def act(self, board: np.ndarray) -> int:
        player = self.current_player(board)
        valid = self.valid_columns(board)

        move = self.winning_move(board, player)
        if move is not None:
            return move

        move = self.winning_move(board, -player)
        if move is not None:
            return move

        root = self.state_key(board, player)
        visits = {}
        values = {}
        state_visits = {}

        for _ in range(self.simulations):
            path = []
            sim_board = board.copy()
            turn = player
            depth = 0

            while depth < self.max_depth and not self.is_final(sim_board):
                actions = self.valid_columns(sim_board)
                key = self.state_key(sim_board, turn)
                unvisited = [action for action in actions if (key, action) not in visits]

                if unvisited:
                    action = unvisited[0]
                    visits[(key, action)] = 0
                    values[(key, action)] = 0.0
                    path.append((key, action))
                    sim_board = self.play_move(sim_board, action, turn)
                    turn = -turn
                    depth += 1
                    break

                action = self.best_uct(key, actions, visits, values, state_visits)
                path.append((key, action))
                sim_board = self.play_move(sim_board, action, turn)
                turn = -turn
                depth += 1

            while depth < self.max_depth and not self.is_final(sim_board):
                actions = self.valid_columns(sim_board)
                action = int(self.rng.choice(actions))
                sim_board = self.play_move(sim_board, action, turn)
                turn = -turn
                depth += 1

            reward = self.reward(sim_board, player)

            for key, action in path:
                state_visits[key] = state_visits.get(key, 0) + 1
                visits[(key, action)] = visits.get((key, action), 0) + 1
                old = values.get((key, action), 0.0)
                count = visits[(key, action)]
                values[(key, action)] = old + (reward - old) / count

        scores = {}
        for action in valid:
            scores[action] = values.get((root, action), -2.0)

        best = max(scores.values())
        best_cols = [col for col in self.center_order if col in scores and scores[col] == best]
        return best_cols[0]

    def best_uct(self, key, actions, visits, values, state_visits) -> int:
        best_action = actions[0]
        best_value = -float("inf")
        total = state_visits.get(key, 1)

        for action in actions:
            count = visits.get((key, action), 1)
            value = values.get((key, action), 0.0)
            uct = value + self.exploration * math.sqrt(math.log(total + 1) / count)
            if uct > best_value:
                best_value = uct
                best_action = action

        return best_action

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

    def is_final(self, board: np.ndarray) -> bool:
        return ConnectState(board).get_winner() != 0 or not any(board[0] == 0)

    def reward(self, board: np.ndarray, player: int) -> float:
        winner = ConnectState(board).get_winner()
        if winner == player:
            return 1.0
        if winner == 0:
            return 0.0
        return -1.0

    def state_key(self, board: np.ndarray, player: int):
        return board.tobytes(), player
