import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connect4.connect_state import ConnectState
from groups.Luis import policy


class RandomPlayer:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def mount(self) -> None:
        pass

    def act(self, board: np.ndarray) -> int:
        valid = [col for col in range(7) if board[0, col] == 0]
        return int(self.rng.choice(valid))


def current_player(board: np.ndarray) -> int:
    red = int(np.sum(board == -1))
    yellow = int(np.sum(board == 1))
    if red == yellow:
        return -1
    return 1


def normalize(features: dict[str, float]) -> dict[str, float]:
    return {
        "bias": features["bias"],
        "center": features["center"] / 6.0,
        "own_two": features["own_two"] / 20.0,
        "own_three": features["own_three"] / 10.0,
        "opp_two": features["opp_two"] / 20.0,
        "opp_three": features["opp_three"] / 10.0,
        "height": features["height"] / 216.0,
    }


def play_training_game(red_agent, yellow_agent, evaluator):
    state = ConnectState()
    history = []
    red_agent.mount()
    yellow_agent.mount()

    while not state.is_final():
        player = state.player
        features = normalize(evaluator.features(state.board, player))
        agent = red_agent if player == -1 else yellow_agent
        col = int(agent.act(state.board))
        history.append((features, player))
        state = state.transition(col)

    return history, state.get_winner()


def update_weights(weights: dict[str, float], history, winner: int, alpha: float) -> dict[str, float]:
    for features, player in history:
        if winner == player:
            target = 1.0
        elif winner == 0:
            target = 0.0
        else:
            target = -1.0

        prediction = 0.0
        for name, value in features.items():
            prediction += weights[name] * value

        error = target - prediction
        for name, value in features.items():
            weights[name] += alpha * error * value

    return weights


def train_random(games: int = 500, alpha: float = 0.04) -> dict[str, float]:
    evaluator = policy.BaseLuisAgent()
    weights = dict(evaluator.weights)
    learner = policy.LuisTactical()
    random_player = RandomPlayer(10)

    for index in range(games):
        evaluator.weights = weights
        if index % 2 == 0:
            history, winner = play_training_game(learner, random_player, evaluator)
        else:
            history, winner = play_training_game(random_player, learner, evaluator)
        weights = update_weights(weights, history, winner, alpha)

    return weights


def train_self_play(games: int = 700, alpha: float = 0.025) -> dict[str, float]:
    random_weights = load_weights("weights_random.json")
    evaluator = policy.BaseLuisAgent()
    evaluator.weights = random_weights
    weights = dict(random_weights)
    learner = policy.LuisValueRandom()
    random_player = RandomPlayer(20)

    for index in range(games):
        evaluator.weights = weights
        learner.weights = weights
        if index % 4 == 0:
            history, winner = play_training_game(learner, random_player, evaluator)
        elif index % 4 == 1:
            history, winner = play_training_game(random_player, learner, evaluator)
        else:
            history, winner = play_training_game(learner, learner, evaluator)
        weights = update_weights(weights, history, winner, alpha)

    return weights


def load_weights(name: str) -> dict[str, float]:
    path = Path(__file__).with_name(name)
    with open(path, "r") as file:
        return json.load(file)


def save_weights(name: str, weights: dict[str, float]) -> None:
    path = Path(__file__).with_name(name)
    with open(path, "w") as file:
        json.dump(weights, file, indent=4)


def main() -> None:
    random_weights = train_random()
    save_weights("weights_random.json", random_weights)
    self_play_weights = train_self_play()
    save_weights("weights_self_play.json", self_play_weights)
    print("Saved weights_random.json")
    print("Saved weights_self_play.json")


if __name__ == "__main__":
    main()
