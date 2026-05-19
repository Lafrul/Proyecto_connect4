# Luis Value Policy Agent

This folder contains three Connect 4 agents:

- `LuisTactical`: wins immediately, blocks immediate losses, and prefers the center.
- `LuisValueRandom`: uses the tactical layer plus a value function trained against a random player.
- `LuisValueSelfPlay`: uses the tactical layer plus a value function trained with random games and self-play.

The final tournament agent is `LuisValueSelfPlay`.

## Main idea

The agent treats Connect 4 as a competitive MDP. Training happens offline. During a real game, the agent does not run Monte Carlo simulations and does not build an MCTS tree. It evaluates each legal next state with a learned value function and chooses the action with the highest value.

The decision rule is:

```text
best_action = argmax V(next_state)
```

Before using the value function, the agent checks two tactical rules:

1. If it can win now, it plays that move.
2. If the opponent can win next, it blocks that move.

## Files

- `policy.py`: agent classes used by the tournament.
- `train.py`: offline training script.
- `weights_random.json`: weights trained against a random player.
- `weights_self_play.json`: weights trained with random games and self-play.
- `gui.py`: graphical evaluator for testing the agents.
- `opponents.py`: local reference versions of Random, Laura MCTS, and Sebastian Trial agents for GUI evaluation.

## Run training

```bash
python groups/Luis/train.py
```

## Run the graphical evaluator

```bash
python groups/Luis/gui.py
```

The GUI can compare the final Luis agent against Random, Laura MCTS, Sebastian Trial, and the three Luis versions.

To play manually, choose `Human` as Red or Yellow, choose the opponent, press `New Game`, and click a column on the board when it is your turn. Each turn has a 60 second timer. If the current player does not move before the timer reaches zero, that player loses.

`Run Simulations` runs the selected number of games only between the two selected automatic agents and writes the result in the table.

## Run the project tournament

```bash
python main.py
```
