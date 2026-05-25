# Agente MCTS — Laura Franco

## Idea principal

Agente basado en **Monte-Carlo Tree Search (MCTS) con selección UCT**. En cada turno construye un árbol de búsqueda corriendo `num_simulations` simulaciones desde el estado actual, balanceando exploración y explotación mediante la fórmula UCT:

```
UCT(s, a) = Q(s,a) + c × √( log(N(s)+1) / (N(s,a)+1) )
```

Adicionalmente, utiliza reglas fijas que buscan evitar amenzas y aprovechar oportunidades específicas.

## Estructura de archivos

```
groups/Laura/
├── policy.py      ← MCTSAgent + HashableConnectState + mcts_uct
├── readme.md      ← este archivo
└── entrega.ipynb  ← análisis empírico completo
```

> Todo el código del agente se encuentra en el archivo 'policy.py' para simplificar los imports.

---

## Parámetros configurables

Atributos de clase en `MCTSAgent` — se pueden cambiar sin tocar el código del agente:

| Parámetro | Default | Efecto |
|---|---|---|
| `num_simulations` | 100 | Más simulaciones = mejor juego, más tiempo por turno |
| `exploration_c` | √2 ≈ 1.41 | Balance exploración/explotación en UCT |

---

## Requisitos

```
python >= 3.10
numpy
math
time
Any, Callable, Dict, Iterable (de typing)
```

No requiere librerías adicionales. El proyecto ya incluye todo lo necesario.

---

## Guía de uso

### Ejecutar el torneo completo

Desde la raíz del proyecto:

```bash
python main.py
```

### Probar el agente manualmente

```python
import numpy as np
from groups.Laura.policy import MCTSAgent

agent = MCTSAgent()
agent.mount()

board = np.zeros((6, 7), dtype=int)

col = agent.act(board)
print(f'El agente juega en la columna: {col}')
```

### Cambiar parámetros configurables
#### Número de simulaciones

```python
agent = MCTSAgent()
agent.mount()
agent.num_simulations = 300
col = agent.act(board)
```

#### Exploration c

```python
agent = MCTSAgent()
agent.mount()
agent.exploration_c = 2
col = agent.act(board)
```

### Correr el análisis del notebook

```bash
cd groups/Laura
jupyter notebook entrega.ipynb
```

El notebook asume que se corre desde la carpeta `groups/Laura/` con el proyecto en el path. Si hay errores de import, ajustar la variable `PROJECT_ROOT` en la celda de setup.

---

## Versiones del agente

| Versión | Descripción | Archivo |
|---|---|---|
| `MCTSAgent` | Versión completa con jugadas forzadas, amenaza doble y rollout inteligente | `policy.py` |
| `MCTSAgentNotForced` | Solo MCTS puro, sin jugadas forzadas ni amenaza doble | definida en `entrega.ipynb` para comparación de estrategias |

La comparación entre versiones se encuentra en la **Sección 4** del notebook.