import sys
import tkinter as tk
import time
from pathlib import Path
from tkinter import ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4.connect_state import ConnectState
from groups.Luis import opponents
from groups.Luis import policy


RandomAgent = opponents.RandomAgent


def play_game(red_agent, yellow_agent) -> int:
    state = ConnectState()
    red_agent.mount()
    yellow_agent.mount()

    while not state.is_final():
        agent = red_agent if state.player == -1 else yellow_agent
        started = time.perf_counter()
        col = int(agent.act(state.board))
        elapsed = time.perf_counter() - started
        if elapsed > 60:
            return 1 if state.player == -1 else -1
        state = state.transition(col)

    return state.get_winner()


def play_match(red_class, yellow_class, games: int) -> tuple[int, int, int]:
    red_wins = 0
    yellow_wins = 0
    draws = 0

    for _ in range(games):
        winner = play_game(red_class(), yellow_class())
        if winner == -1:
            red_wins += 1
        elif winner == 1:
            yellow_wins += 1
        else:
            draws += 1

    return red_wins, yellow_wins, draws


class App:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Luis Connect 4 Lab")
        self.window.geometry("1180x760")
        self.window.minsize(1050, 680)

        self.colors = {
            "bg": "#17202a",
            "panel": "#f7f9fb",
            "board": "#185adb",
            "board_dark": "#0d3f9c",
            "empty": "#edf2f7",
            "red": "#e63946",
            "yellow": "#ffbe0b",
            "text": "#16213e",
            "muted": "#64748b",
            "line": "#d7dee8",
        }

        self.agents = {
            "Random": opponents.RandomAgent,
            "Laura MCTS": opponents.LauraMCTSAgent,
            "Sebastian Trial": opponents.SebastianTrialAgent,
            "Luis Tactical": policy.LuisTactical,
            "Luis Value Random": policy.LuisValueRandom,
            "Luis Value Self Play": policy.LuisValueSelfPlay,
        }
        self.player_names = ["Human"] + list(self.agents.keys())

        self.games = tk.IntVar(value=40)
        self.turn_time = tk.IntVar(value=60)
        self.red_name = tk.StringVar(value="Human")
        self.yellow_name = tk.StringVar(value="Random")
        self.status = tk.StringVar(value="Ready")
        self.score = tk.StringVar(value="Red 0  |  Yellow 0  |  Draws 0")

        self.state = ConnectState()
        self.red_agent = None
        self.yellow_agent = None
        self.red_wins = 0
        self.yellow_wins = 0
        self.draws = 0
        self.move_number = 0
        self.auto_running = False
        self.last_move = None
        self.timer_job = None
        self.turn_started_at = None
        self.turn_limit = 60
        self.auto_delay = 450
        self.timeout_over = False
        self.board_left = 35
        self.board_top = 32
        self.cell = 82
        self.margin = 22

        self.setup_style()
        self.build()
        self.new_game()

    def setup_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["panel"])
        style.configure("Root.TFrame", background=self.colors["bg"])
        style.configure("Title.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 16, "bold"))
        style.configure("Text.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TButton", font=("Segoe UI", 10), padding=7)
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def build(self) -> None:
        root = ttk.Frame(self.window, style="Root.TFrame", padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root, padding=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        header = ttk.Frame(left)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Connect 4 Agent Lab", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, textvariable=self.score, style="Text.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        board_box = ttk.Frame(left, padding=(0, 18, 0, 8))
        board_box.grid(row=1, column=0, sticky="nsew")
        board_box.columnconfigure(0, weight=1)
        board_box.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(board_box, width=700, height=610, highlightthickness=0, bg=self.colors["panel"])
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_board_click)

        self.log = tk.Text(left, height=7, wrap="word", bg="#ffffff", fg=self.colors["text"], relief="flat", padx=10, pady=8, font=("Consolas", 10))
        self.log.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(root, padding=16)
        right.grid(row=0, column=1, sticky="ns")

        ttk.Label(right, text="Agents", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(right, text="Red", style="Text.TLabel").grid(row=1, column=0, sticky="w", pady=(14, 2))
        ttk.Combobox(right, textvariable=self.red_name, values=self.player_names, state="readonly", width=25).grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(right, text="Yellow", style="Text.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(right, textvariable=self.yellow_name, values=self.player_names, state="readonly", width=25).grid(row=4, column=0, columnspan=2, sticky="ew")

        ttk.Label(right, text="Games", style="Text.TLabel").grid(row=5, column=0, sticky="w", pady=(14, 2))
        ttk.Spinbox(right, from_=1, to=500, textvariable=self.games, width=8).grid(row=6, column=0, sticky="w")

        ttk.Label(right, text="Turn timer", style="Text.TLabel").grid(row=5, column=1, sticky="w", pady=(14, 2))
        ttk.Label(right, textvariable=self.turn_time, style="Title.TLabel").grid(row=6, column=1, sticky="w")

        controls = ttk.Frame(right)
        controls.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(18, 10))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ttk.Button(controls, text="New Game", command=self.new_game).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(controls, text="Step", command=self.step_game).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(controls, text="Auto Play", command=self.start_auto, style="Accent.TButton").grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(controls, text="Stop", command=self.stop_auto).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))

        ttk.Button(right, text="Run Match", command=self.run_match).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(right, text="Run Simulations", command=self.run_matrix).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 14))

        columns = ("red", "yellow", "games", "red_wins", "yellow_wins", "draws")
        self.table = ttk.Treeview(right, columns=columns, show="headings", height=9)
        labels = {
            "red": "Red",
            "yellow": "Yellow",
            "games": "Games",
            "red_wins": "R",
            "yellow_wins": "Y",
            "draws": "D",
        }
        widths = {
            "red": 120,
            "yellow": 120,
            "games": 54,
            "red_wins": 42,
            "yellow_wins": 42,
            "draws": 42,
        }
        for col in columns:
            self.table.heading(col, text=labels[col])
            self.table.column(col, width=widths[col], anchor="center")
        self.table.grid(row=10, column=0, columnspan=2, sticky="nsew")

    def new_game(self) -> None:
        self.stop_auto()
        self.stop_timer()
        self.state = ConnectState()
        self.red_agent = self.create_player(self.red_name.get())
        self.yellow_agent = self.create_player(self.yellow_name.get())
        self.move_number = 0
        self.last_move = None
        self.timeout_over = False
        self.log.delete("1.0", "end")
        self.write_log("New game")
        self.write_log("Red: " + self.red_name.get())
        self.write_log("Yellow: " + self.yellow_name.get())
        self.update_status()
        self.draw_board()
        self.start_turn_timer()
        self.continue_human_game()

    def create_player(self, name: str):
        if name == "Human":
            return None
        agent = self.agents[name]()
        agent.mount()
        return agent

    def draw_board(self) -> None:
        self.canvas.delete("all")
        cell = 82
        margin = 22
        width = cell * 7 + margin * 2
        height = cell * 6 + margin * 2
        x0 = 35
        y0 = 32

        self.canvas.create_rectangle(x0 + 8, y0 + 12, x0 + width + 8, y0 + height + 12, fill="#0b2f78", outline="")
        self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, fill=self.colors["board"], outline=self.colors["board_dark"], width=4)

        for col in range(7):
            x = x0 + margin + col * cell + cell / 2
            self.canvas.create_text(x, y0 - 14, text=str(col), fill=self.colors["text"], font=("Segoe UI", 12, "bold"))

        for row in range(6):
            for col in range(7):
                x = x0 + margin + col * cell + cell / 2
                y = y0 + margin + row * cell + cell / 2
                value = int(self.state.board[row, col])
                fill = self.colors["empty"]
                outline = "#cbd5e1"
                if value == -1:
                    fill = self.colors["red"]
                    outline = "#9f1d2b"
                elif value == 1:
                    fill = self.colors["yellow"]
                    outline = "#b7791f"
                self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30, fill=fill, outline=outline, width=3)
                if self.last_move == (row, col):
                    self.canvas.create_oval(x - 37, y - 37, x + 37, y + 37, outline="#ffffff", width=4)

        turn_color = "Red" if self.state.player == -1 else "Yellow"
        turn_name = self.red_name.get() if self.state.player == -1 else self.yellow_name.get()
        message = "Turn: " + turn_color
        if turn_name == "Human" and not self.state.is_final():
            message += " - click a column"
        self.canvas.create_text(335, 555, text=message, fill=self.colors["text"], font=("Segoe UI", 15, "bold"))

    def update_status(self) -> None:
        if self.timeout_over:
            self.score.set("Red " + str(self.red_wins) + "  |  Yellow " + str(self.yellow_wins) + "  |  Draws " + str(self.draws))
            return
        if self.state.is_final():
            winner = self.state.get_winner()
            if winner == -1:
                text = "Red wins"
            elif winner == 1:
                text = "Yellow wins"
            else:
                text = "Draw"
            self.status.set(text)
        else:
            name = self.red_name.get() if self.state.player == -1 else self.yellow_name.get()
            color = "Red" if self.state.player == -1 else "Yellow"
            if name == "Human":
                self.status.set(color + " to move: click a column")
            else:
                self.status.set(color + " to move: " + name)
        self.score.set("Red " + str(self.red_wins) + "  |  Yellow " + str(self.yellow_wins) + "  |  Draws " + str(self.draws))

    def step_game(self) -> None:
        if self.timeout_over:
            return
        if self.state.is_final():
            self.finish_game()
            return

        agent = self.red_agent if self.state.player == -1 else self.yellow_agent
        if agent is None:
            self.update_status()
            return

        started = time.perf_counter()
        col = int(agent.act(self.state.board))
        elapsed = time.perf_counter() - started
        if elapsed > self.turn_limit:
            self.finish_timeout()
            return
        self.apply_move(col)

    def on_board_click(self, event) -> None:
        if self.timeout_over or self.state.is_final():
            return
        agent = self.red_agent if self.state.player == -1 else self.yellow_agent
        if agent is not None:
            return
        col = self.column_from_x(event.x)
        if col is None:
            return
        if not self.state.is_applicable(col):
            self.write_log("Column " + str(col) + " is not available")
            return
        if self.turn_started_at is not None and time.perf_counter() - self.turn_started_at > self.turn_limit:
            self.finish_timeout()
            return
        self.apply_move(col)
        if self.auto_running and not self.state.is_final():
            self.window.after(self.auto_delay, self.auto_step)
        else:
            self.continue_human_game()

    def continue_human_game(self) -> None:
        if self.timeout_over or self.state.is_final():
            return
        has_human = self.red_name.get() == "Human" or self.yellow_name.get() == "Human"
        agent = self.red_agent if self.state.player == -1 else self.yellow_agent
        if has_human and agent is not None:
            self.window.after(self.auto_delay, self.auto_step)

    def column_from_x(self, x: int) -> int | None:
        board_x = x - self.board_left - self.margin
        if board_x < 0:
            return None
        col = int(board_x // self.cell)
        if 0 <= col < 7:
            return col
        return None

    def apply_move(self, col: int) -> None:
        self.stop_timer()
        color = "Red" if self.state.player == -1 else "Yellow"
        name = self.red_name.get() if self.state.player == -1 else self.yellow_name.get()
        row = self.find_row(self.state.board, col)
        self.state = self.state.transition(col)
        self.last_move = (row, col)
        self.move_number += 1
        self.write_log(str(self.move_number) + ". " + color + " (" + name + ") plays column " + str(col))
        self.draw_board()
        self.update_status()

        if self.state.is_final():
            self.finish_game()
        else:
            self.start_turn_timer()

    def find_row(self, board: np.ndarray, col: int) -> int:
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row
        return 0

    def finish_game(self) -> None:
        self.stop_timer()
        winner = self.state.get_winner()
        if winner == -1:
            self.red_wins += 1
            self.write_log("Result: Red wins")
        elif winner == 1:
            self.yellow_wins += 1
            self.write_log("Result: Yellow wins")
        else:
            self.draws += 1
            self.write_log("Result: Draw")
        self.update_status()
        self.stop_auto()

    def start_turn_timer(self) -> None:
        self.stop_timer()
        if self.timeout_over or self.state.is_final():
            return
        self.turn_started_at = time.perf_counter()
        self.turn_time.set(self.turn_limit)
        self.tick_timer()

    def stop_timer(self) -> None:
        if self.timer_job is not None:
            self.window.after_cancel(self.timer_job)
            self.timer_job = None

    def tick_timer(self) -> None:
        if self.timeout_over or self.state.is_final() or self.turn_started_at is None:
            return
        elapsed = int(time.perf_counter() - self.turn_started_at)
        left = self.turn_limit - elapsed
        if left <= 0:
            self.turn_time.set(0)
            self.finish_timeout()
            return
        self.turn_time.set(left)
        self.timer_job = self.window.after(250, self.tick_timer)

    def finish_timeout(self) -> None:
        if self.timeout_over:
            return
        self.stop_timer()
        self.timeout_over = True
        late_color = "Red" if self.state.player == -1 else "Yellow"
        if self.state.player == -1:
            self.yellow_wins += 1
            winner = "Yellow"
        else:
            self.red_wins += 1
            winner = "Red"
        self.write_log("Result: " + late_color + " lost by timeout")
        self.status.set(winner + " wins by timeout")
        self.score.set("Red " + str(self.red_wins) + "  |  Yellow " + str(self.yellow_wins) + "  |  Draws " + str(self.draws))
        self.stop_auto()

    def start_auto(self) -> None:
        if self.timeout_over or self.state.is_final():
            self.new_game()
        self.auto_running = True
        self.auto_step()

    def auto_step(self) -> None:
        if not self.auto_running:
            return
        if self.timeout_over:
            return
        if self.state.is_final():
            self.finish_game()
            return
        agent = self.red_agent if self.state.player == -1 else self.yellow_agent
        if agent is None:
            self.update_status()
            return
        self.step_game()
        if self.auto_running:
            self.window.after(self.auto_delay, self.auto_step)

    def stop_auto(self) -> None:
        self.auto_running = False

    def write_log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def clear_table(self) -> None:
        for row in self.table.get_children():
            self.table.delete(row)

    def add_result(self, red: str, yellow: str, games: int, red_wins: int, yellow_wins: int, draws: int) -> None:
        self.table.insert("", "end", values=(red, yellow, games, red_wins, yellow_wins, draws))

    def run_match(self) -> None:
        self.stop_auto()
        self.clear_table()
        red = self.red_name.get()
        yellow = self.yellow_name.get()
        if red == "Human" or yellow == "Human":
            self.status.set("Run Match only works with automatic agents")
            self.write_log("Choose automatic agents for Run Match")
            return
        games = self.games.get()
        red_wins, yellow_wins, draws = play_match(self.agents[red], self.agents[yellow], games)
        self.add_result(red, yellow, games, red_wins, yellow_wins, draws)
        self.status.set("Finished match: " + red + " vs " + yellow)

    def run_matrix(self) -> None:
        self.stop_auto()
        self.clear_table()
        red = self.red_name.get()
        yellow = self.yellow_name.get()
        if red == "Human" or yellow == "Human":
            self.status.set("Run Simulations only works with automatic agents")
            self.write_log("Choose automatic agents for Run Simulations")
            return
        games = self.games.get()
        red_wins, yellow_wins, draws = play_match(self.agents[red], self.agents[yellow], games)
        self.add_result(red, yellow, games, red_wins, yellow_wins, draws)
        self.status.set("Finished simulations: " + red + " vs " + yellow)

    def run(self) -> None:
        self.window.mainloop()


if __name__ == "__main__":
    App().run()
