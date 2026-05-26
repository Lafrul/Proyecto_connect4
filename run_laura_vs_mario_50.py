import importlib.util
import os
import time
from pathlib import Path

import pandas as pd

from connect4.connect_state import ConnectState


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "laura_vs_mario_50_results.csv"
TARGET_GAMES = int(os.environ.get("TARGET_GAMES", "50"))
MAX_NEW_GAMES = int(os.environ.get("MAX_NEW_GAMES", "0"))

LAURA_NAME = "Agente Laura MCTS"
MARIO_NAME = "Agente Mario"


def load_class(relative_path: str, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, class_name)


LauraPolicy = load_class(
    "groups/Group laura/policy.py", "group_laura_policy_mario_runner", "MCTSAgent"
)
MarioPolicy = load_class(
    "groups/Group mario/policy.py", "group_mario_policy_laura_runner", "Winner"
)


def load_existing() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(OUTPUT_PATH)


def completed_game_ids(df: pd.DataFrame) -> set[int]:
    if df.empty or "game_id" not in df.columns:
        return set()
    return set(df["game_id"].dropna().astype(int).unique())


def budget_for(agent_name: str) -> int:
    if agent_name == LAURA_NAME:
        return int(getattr(LauraPolicy, "num_simulations", 0))
    return 0


def play_one(game_id: int) -> list[dict]:
    laura_starts = game_id % 2 == 0
    if laura_starts:
        first_name, first_cls = LAURA_NAME, LauraPolicy
        second_name, second_cls = MARIO_NAME, MarioPolicy
    else:
        first_name, first_cls = MARIO_NAME, MarioPolicy
        second_name, second_cls = LAURA_NAME, LauraPolicy

    first = first_cls()
    second = second_cls()
    first.mount()
    second.mount()

    state = ConnectState()
    moves = 0
    decision_time = {first_name: 0.0, second_name: 0.0}
    decisions = {first_name: 0, second_name: 0}
    start = time.perf_counter()

    while not state.is_final():
        current_name = first_name if state.player == -1 else second_name
        current_policy = first if state.player == -1 else second

        t0 = time.perf_counter()
        action = int(current_policy.act(state.board))
        elapsed = time.perf_counter() - t0

        decision_time[current_name] += elapsed
        decisions[current_name] += 1

        state = state.transition(action)
        moves += 1

    total_game_time = time.perf_counter() - start
    winner_token = int(state.get_winner())
    if winner_token == -1:
        winner_agent = first_name
    elif winner_token == 1:
        winner_agent = second_name
    else:
        winner_agent = "Draw"

    rows = []
    for focal_name, focal_side, focal_color, opponent_name in [
        (first_name, "agent_1", "first", second_name),
        (second_name, "agent_2", "second", first_name),
    ]:
        if winner_agent == "Draw":
            result = "draw"
            score = 0
        elif winner_agent == focal_name:
            result = "win"
            score = 1
        else:
            result = "loss"
            score = -1

        rows.append(
            {
                "experiment_id": "Laura_vs_Mario_alternating_50",
                "match_id": f"Laura_vs_Mario_game_{game_id:03d}",
                "agent_1": first_name,
                "agent_2": second_name,
                "focal_agent": focal_name,
                "focal_side": focal_side,
                "opponent": opponent_name,
                "focal_color": focal_color,
                "trials": budget_for(focal_name),
                "opponent_trials": budget_for(opponent_name),
                "configuration_trials": budget_for(focal_name),
                "game_id": game_id,
                "seed": 2026 + game_id,
                "starter": first_name,
                "laura_simulations": int(getattr(LauraPolicy, "num_simulations", 0)),
                "mario_agent": MarioPolicy.__name__,
                "winner": "agent_1"
                if winner_agent == first_name
                else "agent_2"
                if winner_agent == second_name
                else "draw",
                "winner_agent": winner_agent,
                "result": result,
                "score": score,
                "num_moves": moves,
                "total_game_time": total_game_time,
                "mean_decision_time": decision_time[focal_name]
                / max(1, decisions[focal_name]),
                "total_decision_time": decision_time[focal_name],
                "num_decisions": decisions[focal_name],
            }
        )

    return rows


def main() -> None:
    existing = load_existing()
    done = completed_game_ids(existing)
    start_time = time.perf_counter()
    new_games = 0

    for game_id in range(TARGET_GAMES):
        if game_id in done:
            continue
        if MAX_NEW_GAMES and new_games >= MAX_NEW_GAMES:
            break

        rows = play_one(game_id)
        new_df = pd.DataFrame(rows)
        if OUTPUT_PATH.exists():
            new_df.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
        else:
            new_df.to_csv(OUTPUT_PATH, index=False)

        winner = rows[0]["winner_agent"]
        elapsed = time.perf_counter() - start_time
        done.add(game_id)
        new_games += 1
        print(
            f"game {game_id + 1:02d}/{TARGET_GAMES} | "
            f"starter={rows[0]['starter']} | winner={winner} | "
            f"moves={rows[0]['num_moves']} | "
            f"time={rows[0]['total_game_time']:.2f}s | "
            f"elapsed={elapsed / 60:.1f}min",
            flush=True,
        )

    print(f"CSV guardado en: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
