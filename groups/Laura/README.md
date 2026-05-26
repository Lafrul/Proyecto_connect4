# Agente MCTS — Laura Franco

## Estructura de archivos

```
groups/Laura/
├── policy.py      ← MCTSAgent + HashableConnectState + mcts_uct
├── readme.md      ← este archivo
└── entrega.ipynb  ← análisis empírico completo
```

> Todo el código del agente se encuentra en el archivo 'policy.py' para simplificar los imports.

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