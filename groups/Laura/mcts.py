import math
from typing import Any, Callable, Dict, Iterable
import numpy as np

def mcts_uct(
    root_state: Any,
    legal_actions_fn: Callable[[Any], Iterable[Any]],
    successor_fn: Callable[[Any, Any, np.random.RandomState], Any],
    terminal_fn: Callable[[Any], bool],
    reward_fn: Callable[[Any], float],
    *,
    num_simulations: int,
    max_depth: int,
    exploration_c: float,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """
    Run a minimal single-agent Monte-Carlo Tree Search (MCTS) with UCT selection.

    This is a generic, environment-agnostic implementation of MCTS using
    the UCT (Upper Confidence bounds applied to Trees) rule for selecting
    actions during tree traversal. It assumes:

    - A single decision-making agent.
    - An episodic environment with finite horizon.
    - Rewards are obtained only at terminal states.
    - Discount factor :math:`\\gamma = 1` (no discounting).

    The function maintains and updates three sets of statistics:

    - ``N_s[s]``: state visit counts.
    - ``N_sa[(s, a)]``: state-action visit counts.
    - ``Q_sa[(s, a)]``: estimated mean return (value) for each state-action pair.

    For each simulation, it performs:

    1. **Selection + Expansion**

       Starting at ``root_state``, repeatedly:

       - If the current state is terminal, or ``max_depth`` is reached,
         or there are no legal actions, stop selection.
       - Otherwise, check for **unvisited** actions at this state:
         actions ``a`` such that ``(s, a)`` is not in ``N_sa``.
         If any exist, *expand* the first unvisited action (in the
         deterministic order given by ``list(legal_actions_fn(s))``).
       - If all actions have been visited, select an action using UCT:

         .. math::

             \\text{UCT}(s, a) = Q_{sa}(s, a) +
                 c \\sqrt{\\frac{\\log(N_s(s) + 1)}{N_{sa}(s, a) + 1}},

         where ``c`` is ``exploration_c``.

       At each visited (state, action) pair in this selection/expansion
       phase the function:

       - Appends ``(s, a)`` to the simulation path.
       - Increments ``N_s[s]`` and ``N_sa[(s, a)]``.
       - Steps to the next state via ``successor_fn``.

    2. **Rollout**

       From the state reached after selection/expansion, the algorithm
       follows a *uniform random policy* until:

       - A terminal state is reached, or
       - The total depth reaches ``max_depth``.

       Random actions are sampled using the provided ``rng``.

    3. **Backpropagation**

       Let ``R`` be the reward of the final state if it is terminal,
       and ``0.0`` otherwise (truncated rollout). For every
       ``(s, a)`` on the recorded path, the Q-value estimate is updated
       as an incremental mean:

       .. math::

           Q_{sa}(s, a) \\leftarrow
             Q_{sa}(s, a) + \\frac{R - Q_{sa}(s, a)}{N_{sa}(s, a)}.

    After all simulations, the function extracts from the root state:

    - ``q_root[a]``: the estimated Q-value for each root action.
    - ``n_root[a]``: the visit count for each root action.

    It also computes a recommended ``best_action`` based on the largest
    Q-value in ``q_root``, breaking ties deterministically.

    Parameters
    ----------
    root_state : hashable
        Starting state for planning. This is the root of the MCTS tree.
        The state must be hashable so that it can be used as a key in
        Python dictionaries.
    legal_actions_fn : callable
        Function ``legal_actions_fn(s) -> iterable`` returning the legal
        actions available in state ``s``.

        - The returned iterable will be converted to a list each time it
          is used, and that list order is used for deterministic tie
          breaking in UCT and for sampling during rollout.
        - It is assumed to return an empty iterable for terminal states.
    successor_fn : callable
        Function ``successor_fn(s, a, rng) -> s'`` that returns the next
        state when action ``a`` is applied in state ``s``.

        - The function *must* use the provided ``rng`` for any internal
          randomness and should not rely on global random state.
        - The tests assume the environment is finite and acyclic, so
          repeated application of successor_fn will eventually terminate.
    terminal_fn : callable
        Function ``terminal_fn(s) -> bool`` that returns ``True`` if
        the state is terminal and ``False`` otherwise.
    reward_fn : callable
        Function ``reward_fn(s) -> float`` that returns the reward of a
        *terminal* state.

        - The function is only called on states for which
          ``terminal_fn(s)`` is ``True`` (in correct usage).
        - Non-terminal states are never passed to ``reward_fn``.
    num_simulations : int
        Number of MCTS simulations (episodes) to run. Each simulation
        performs its own selection, expansion, rollout, and
        backpropagation starting at ``root_state``.
    max_depth : int
        Maximum allowed depth for each simulation (including both the
        tree part and the random rollout). If this depth is reached
        without reaching a terminal state, the rollout is truncated and
        the return is taken as ``0.0``.
    exploration_c : float
        Exploration constant :math:`c` used in the UCT formula.
        Larger values lead to more exploration; smaller values lead to
        more exploitation of currently high-valued actions.
    rng : numpy.random.RandomState
        Random number generator used during:

        - selection/expansion (through calls to ``successor_fn``), and
        - rollout (uniform random action selection).

        The global NumPy random state is never used inside this function.

    Returns
    -------
    result : dict
        A dictionary with the following keys:

        ``'q_root'`` : dict[Any, float]
            Estimated mean return for each *root* action that was
            visited at least once:

            .. code-block:: python

                q_root[a] = Q_sa[(root_state, a)]

            for each action ``a`` in ``legal_actions_fn(root_state)``
            that has been explored.
        ``'n_root'`` : dict[Any, int]
            Visit counts at the root for each explored action:

            .. code-block:: python

                n_root[a] = N_sa[(root_state, a)]

        ``'best_action'`` : Any or None
            The action in ``q_root`` with the largest Q-value. Ties are
            broken deterministically by sorting the actions (using the
            default Python ordering) and taking the first one with the
            maximum value.

            If there are no explored actions at the root (for example if
            the root is terminal or if there are zero legal actions),
            ``best_action`` is ``None``.

    """
    N_s = {}
    N_sa = {}
    Q_sa = {}

    for _ in range(num_simulations):
        s = root_state
        depth = 0
        path = []

        # Selection + Expansion
        while not terminal_fn(s) and depth < max_depth and len(legal_actions_fn(s)) != 0:
            actions = legal_actions_fn(s)
            unvisited_a = [a for a in actions if (s,a) not in N_sa]

            # Si todavía hay acciones sin visitar se visitan
            if unvisited_a:
                a = unvisited_a[0]
                path.append((s,a))

                N_s[s] = N_s.get(s, 0) + 1
                N_sa[(s,a)] = 1
                Q_sa[(s,a)] = 0

                s = successor_fn(s, a, rng)
                depth += 1
                break   # para que solo expanda hasta añadir UN SOLO nuevo nodo al arbol

            # Si ya se han visitado todas las acciones se ecoge una con UCT (explore, exploit)
            else:
                best_a = None
                best_value = -float("inf")

                for a in actions:
                    uct = Q_sa[(s,a)] + exploration_c * math.sqrt(math.log(N_s.get(s,0) + 1) / (N_sa[(s,a)] + 1))
                    if uct > best_value:
                        best_value = uct
                        best_a = a
                
                a = best_a
                path.append((s,a))

                N_s[s] = N_s.get(s,0) + 1
                N_sa[(s,a)] += 1

                s = successor_fn(s, a, rng)
                depth += 1

        # Rollout
        while not terminal_fn(s) and depth < max_depth and len(legal_actions_fn(s)) != 0:
            actions = legal_actions_fn(s)

            a = rng.choice(actions)
            s = successor_fn(s, a, rng)
            depth += 1

        # Rewards
        if terminal_fn(s):
            R = reward_fn(s)
        else:
            R = 0
        
        # Backpropagation: solo guarda/actualiza los valores de los nodos que se visitan en la parte de selection + expansion
        for (state, action) in path:
            Q_sa[(state, action)] = Q_sa[(state, action)] + (R - Q_sa[(state, action)]) / N_sa[(state, action)]

    # Returns
    actions = legal_actions_fn(root_state)

    q_root = {}
    n_root = {}

    for a in actions:
        if (root_state, a) in N_sa:
            q_root[a] = Q_sa[(root_state, a)]
            n_root[a] = N_sa[(root_state, a)]

    best_action = None

    if q_root:
        max_q = max(q_root.values())
        best_candidates = [a for a in q_root if q_root[a] == max_q]
        best_action = sorted(best_candidates)[0]

    return {"q_root": q_root, "n_root": n_root, "best_action": best_action,}

        