import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

try:
    from typing import override
except ImportError:
    def override(func):
        return func


class HeuristicPolicyC4(Policy):
    WINNING_LENGTH = 4

    WIN_SCORE = 1_000_000.0
    DIRECT_BLOCK_SCORE = 120_000.0

    DOUBLE_THREAT_SCORE = 25_000.0
    SINGLE_THREAT_SCORE = 4_000.0

    OPPONENT_NEXT_WIN_PENALTY = 500_000.0

    POSITION_WEIGHTS = np.array(
        [
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3],
        ],
        dtype=float,
    )

    def __init__(self, rows: int = ConnectState.ROWS, cols: int = ConnectState.COLS):
        self.rows = rows
        self.cols = cols

        self.all_windows = self._build_all_windows()
        self.windows_by_cell = self._build_windows_by_cell()

    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        player = self._current_player(s)
        return self.get_action(s, player)

    def get_action(
        self, board: np.ndarray, player: int, return_scores: bool = False
    ) -> int | tuple[int, np.ndarray]:
        valid_moves = self._get_valid_moves(board)

        if len(valid_moves) == 0:
            raise ValueError("No valid moves available")

        opponent = -player

        # Antes esto se recalculaba dentro de cada columna.
        # Ahora se calcula una sola vez por decisión.
        opponent_wins_now = set(self._get_winning_moves_inplace(board, opponent))

        scores = np.full(self.cols, -np.inf, dtype=float)

        for col in valid_moves:
            scores[col] = self._evaluate_move(
                board=board,
                col=col,
                player=player,
                opponent_wins_now=opponent_wins_now,
            )

        best_score = float(np.max(scores[valid_moves]))
        best_moves = [col for col in valid_moves if scores[col] == best_score]

        center = (self.cols - 1) / 2.0
        best_action = int(min(best_moves, key=lambda c: (abs(c - center), c)))

        if return_scores:
            return best_action, scores

        return best_action

    def _evaluate_move(
        self,
        board: np.ndarray,
        col: int,
        player: int,
        opponent_wins_now: set[int],
    ) -> float:
        opponent = -player

        board_after = board.copy()
        row = self._drop_piece(board_after, col, player)

        if row < 0:
            return -np.inf

        # 1. Ganar inmediatamente.
        if self._check_win(board_after, row, col, player):
            return self.WIN_SCORE

        score = 0.0

        # 2. Bloquear victoria inmediata.
        if col in opponent_wins_now:
            score += self.DIRECT_BLOCK_SCORE

        # 3. Posición local.
        score += self._position_advantage(row, col)

        # 4. Ventanas locales alrededor de la ficha nueva.
        score += self._local_window_score(board_after, row, col, player)

        # 5. Bloqueos locales de ventanas del rival.
        score += self._block_threats_local(board, row, col, opponent)

        # 6. Bonus suave de movilidad.
        score += self._mobility_bonus(board_after, col)

        # 7. Amenazas futuras. Esto es más costoso, pero sigue siendo barato
        # porque usa drop + undo en vez de copiar el tablero muchas veces.
        own_next_wins = self._get_winning_moves_inplace(board_after, player)
        opponent_next_wins = self._get_winning_moves_inplace(board_after, opponent)

        if len(own_next_wins) >= 2:
            score += self.DOUBLE_THREAT_SCORE * len(own_next_wins)
        elif len(own_next_wins) == 1:
            score += self.SINGLE_THREAT_SCORE

        if opponent_next_wins:
            score -= self.OPPONENT_NEXT_WIN_PENALTY * len(opponent_next_wins)

        return score

    def _drop_piece(self, board: np.ndarray, col: int, player: int) -> int:
        for row in range(self.rows - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row

        return -1

    def _undo_piece(self, board: np.ndarray, row: int, col: int) -> None:
        if row >= 0:
            board[row, col] = 0

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        if row < 0:
            return False

        directions = [
            (0, 1),    # horizontal
            (1, 0),    # vertical
            (1, 1),    # diagonal \
            (1, -1),   # diagonal /
        ]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= self.WINNING_LENGTH:
                return True

        return False

    def _get_valid_moves(self, board: np.ndarray) -> list[int]:
        return [col for col in range(self.cols) if board[0, col] == 0]

    def _get_winning_moves_inplace(self, board: np.ndarray, player: int) -> list[int]:
        """
        Busca columnas ganadoras sin hacer board.copy() por cada columna.
        Hace drop + check + undo.
        """
        winning_moves = []

        for col in self._get_valid_moves(board):
            row = self._drop_piece(board, col, player)

            if row >= 0 and self._check_win(board, row, col, player):
                winning_moves.append(col)

            self._undo_piece(board, row, col)

        return winning_moves

    def _position_advantage(self, row: int, col: int) -> float:
        return float(self.POSITION_WEIGHTS[row, col])

    def _local_window_score(
        self, board: np.ndarray, row: int, col: int, player: int
    ) -> float:
        """
        Evalúa solo las ventanas de 4 que pasan por la ficha nueva.
        Mucho más rápido que recorrer todas las ventanas del tablero.
        """
        score = 0.0
        opponent = -player

        for window in self.windows_by_cell[row][col]:
            values = [board[r, c] for r, c in window]

            player_count = values.count(player)
            opponent_count = values.count(opponent)
            empty_cells = [(r, c) for r, c in window if board[r, c] == 0]
            empty_count = len(empty_cells)

            if player_count > 0 and opponent_count > 0:
                continue

            playable_empty_count = 0
            for r, c in empty_cells:
                if self._is_playable_empty(board, r, c):
                    playable_empty_count += 1

            if opponent_count == 0:
                if player_count == 3 and empty_count == 1:
                    score += 2_800.0 if playable_empty_count else 450.0
                elif player_count == 2 and empty_count == 2:
                    score += 160.0 + 50.0 * playable_empty_count
                elif player_count == 1 and empty_count == 3:
                    score += 12.0

            elif player_count == 0:
                if opponent_count == 3 and empty_count == 1:
                    score -= 3_200.0 if playable_empty_count else 500.0
                elif opponent_count == 2 and empty_count == 2:
                    score -= 180.0 + 60.0 * playable_empty_count
                elif opponent_count == 1 and empty_count == 3:
                    score -= 10.0

        return score

    def _block_threats_local(
        self, original_board: np.ndarray, row: int, col: int, opponent: int
    ) -> float:
        """
        Premia ocupar una celda que hacía parte de una ventana prometedora del rival.
        Usa solo ventanas que pasan por la celda jugada.
        """
        if original_board[row, col] != 0:
            return 0.0

        score = 0.0

        for window in self.windows_by_cell[row][col]:
            values = [original_board[r, c] for r, c in window]

            opponent_count = values.count(opponent)
            empty_cells = [(r, c) for r, c in window if original_board[r, c] == 0]
            empty_count = len(empty_cells)

            if opponent_count == 3 and empty_count == 1:
                score += 1_700.0
            elif opponent_count == 2 and empty_count == 2:
                score += 180.0
            elif opponent_count == 1 and empty_count == 3:
                score += 15.0

        return score

    def _mobility_bonus(self, board: np.ndarray, col: int) -> float:
        neighbors_free = 0

        for adj_col in range(max(0, col - 1), min(self.cols, col + 2)):
            if board[0, adj_col] == 0:
                neighbors_free += 1

        return 4.0 * neighbors_free

    def _is_playable_empty(self, board: np.ndarray, row: int, col: int) -> bool:
        return board[row, col] == 0 and (
            row == self.rows - 1 or board[row + 1, col] != 0
        )

    def _build_all_windows(self) -> list[list[tuple[int, int]]]:
        windows = []
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]

        for row in range(self.rows):
            for col in range(self.cols):
                for dr, dc in directions:
                    window = []

                    for k in range(self.WINNING_LENGTH):
                        r = row + k * dr
                        c = col + k * dc

                        if 0 <= r < self.rows and 0 <= c < self.cols:
                            window.append((r, c))

                    if len(window) == self.WINNING_LENGTH:
                        windows.append(window)

        return windows

    def _build_windows_by_cell(self) -> list[list[list[list[tuple[int, int]]]]]:
        windows_by_cell = [
            [[] for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        for window in self.all_windows:
            for r, c in window:
                windows_by_cell[r][c].append(window)

        return windows_by_cell

    def get_heuristic_features(
        self, board: np.ndarray, player: int
    ) -> dict[str, int | list[int]]:
        """
        Esta función es solo para análisis/debug. No debería usarse dentro de los rollouts.
        """
        opponent = -player

        features: dict[str, int | list[int]] = {
            "valid_moves": self._get_valid_moves(board),
            "player_open_2": 0,
            "player_open_3": 0,
            "player_playable_3": 0,
            "opponent_open_2": 0,
            "opponent_open_3": 0,
            "opponent_playable_3": 0,
            "center_occupancy_player": int(np.sum(board[:, self.cols // 2] == player)),
            "center_occupancy_opponent": int(
                np.sum(board[:, self.cols // 2] == opponent)
            ),
        }

        for window in self.all_windows:
            values = [board[r, c] for r, c in window]

            player_count = values.count(player)
            opponent_count = values.count(opponent)
            empty_cells = [(r, c) for r, c in window if board[r, c] == 0]
            empty_count = len(empty_cells)

            if opponent_count == 0:
                if player_count == 2 and empty_count == 2:
                    features["player_open_2"] += 1
                elif player_count == 3 and empty_count == 1:
                    features["player_open_3"] += 1
                    if any(self._is_playable_empty(board, r, c) for r, c in empty_cells):
                        features["player_playable_3"] += 1

            if player_count == 0:
                if opponent_count == 2 and empty_count == 2:
                    features["opponent_open_2"] += 1
                elif opponent_count == 3 and empty_count == 1:
                    features["opponent_open_3"] += 1
                    if any(self._is_playable_empty(board, r, c) for r, c in empty_cells):
                        features["opponent_playable_3"] += 1

        return features

    @staticmethod
    def _current_player(board: np.ndarray) -> int:
        red_moves = int(np.sum(board == -1))
        yellow_moves = int(np.sum(board == 1))
        return -1 if red_moves == yellow_moves else 1