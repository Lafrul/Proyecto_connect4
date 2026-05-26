# Connect-4: Trial-Based Online Policy Improvement

Este proyecto implementa y evalua un agente para Connect-4 basado en **Trial-Based Online Policy Improvement**. La version final usa rollouts heuristicos para mejorar la calidad de las simulaciones usadas al escoger cada jugada.

## Archivos principales

- `groups/Group S/policy.py`: politica final del agente.
- `groups/Group S/Euristic.py`: heuristica usada durante los rollouts.
- `groups/Group J/policy.py`: version base con rollouts aleatorios.
- `entrega.ipynb`: notebook de entrega con el estudio experimental y las graficas.
- `resultados_experimentos.csv`: datos usados para el analisis principal.
- `figuras_entrega/`: graficas exportadas para la entrega.
- `generar_graficas_entrega.py`: script para regenerar las graficas principales desde el CSV.

## Idea del agente

El agente evalua las acciones legales mediante simulaciones completas. Para cada columna disponible, aplica la accion candidata, ejecuta varios trials desde ese estado y estima el valor promedio de la jugada.

La version base usa rollouts aleatorios. La version final mantiene la misma estructura de Trial-Based Online Policy Improvement, pero reemplaza los rollouts aleatorios por una politica heuristica que prioriza:

- ganar inmediatamente si existe la oportunidad;
- bloquear victorias inmediatas del rival;
- controlar columnas centrales;
- crear amenazas;
- evitar movimientos que entreguen una victoria directa al oponente.

## Experimentos

Se evaluo el rendimiento como funcion de:

- cantidad de trials;
- tipo de rollout: aleatorio o heuristico;
- oponente;
- orden de juego: primero o segundo.

Los escenarios principales son:

- version con rollouts aleatorios vs `Random`;
- version con rollouts heuristicos vs `Random`;
- version con rollouts heuristicos vs version con rollouts aleatorios;
- self-play de cada version.

## Resultados principales

La version con rollouts heuristicos gana el 100% de las partidas contra `Random` en los trials evaluados. La version con rollouts aleatorios tambien alcanza 100%, pero necesita mas trials para estabilizarse.

En la comparacion directa, la version heuristica obtiene score positivo frente a la version con rollouts aleatorios en todos los presupuestos agregados. Esto sugiere que la heuristica mejora la calidad de las simulaciones, no solo el desempeno contra un jugador aleatorio.

La principal debilidad es el costo computacional: cada rollout heuristico tarda mas que un rollout aleatorio, por lo que aumentar mucho los trials incrementa el tiempo por jugada.

## Como usar el agente

El agente final esta en `groups/Group S/policy.py` y la clase que debe cargarse es `Winner`.

Para usarlo dentro del torneo del proyecto, incluye el participante de Group S en la lista de jugadores usando el mismo mecanismo de carga que usa el torneo. Si se necesita cargarlo manualmente, funciona con `importlib`:

```python
import importlib

Winner = importlib.import_module("groups.Group S.policy").Winner

players = [
    ("Group S", Winner),
]
```

La politica recibe el tablero actual como un `np.ndarray` de 6x7 y retorna la columna elegida:

```python
agent = Winner()
agent.mount()
columna = agent.act(board)
```

Por defecto usa `50` trials por accion. Ese valor se puede ajustar en `Winner.TRIALS` si se quiere priorizar velocidad o calidad de decision.

## Mejoras futuras

- Usar asignacion adaptativa de trials para no gastar simulaciones en acciones claramente inferiores.
- Cortar temprano estados tacticamente obvios, como victorias o bloqueos forzados.
- Mantener diversidad en los rollouts heuristicos mediante seleccion estocastica entre buenas acciones.
- Aumentar partidas en configuraciones con mayor varianza experimental.
