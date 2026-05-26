from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "s_vs_mario_50_results.csv"
OUT_DIR = ROOT / "figuras_entrega"
OUT_DIR.mkdir(exist_ok=True)

S_NAME = "Agente con rollouts heuristicos"
MARIO_NAME = "Agente Mario"

COLORS = {
    S_NAME: "#1b9e77",
    MARIO_NAME: "#8a5a44",
    "Draw": "#d9a441",
}


def setup(ax):
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def save(fig, filename: str):
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


df = pd.read_csv(CSV_PATH)
games = (
    df[df["focal_side"] == "agent_1"]
    .copy()
    .sort_values("game_id")
    .reset_index(drop=True)
)

summary_rows = []
for starter, group in [("Todos", games), *games.groupby("starter")]:
    total = len(group)
    wins_s = int((group["winner_agent"] == S_NAME).sum())
    wins_mario = int((group["winner_agent"] == MARIO_NAME).sum())
    draws = int((group["winner_agent"] == "Draw").sum())
    summary_rows.append(
        {
            "starter": starter,
            "games": total,
            "s_wins": wins_s,
            "mario_wins": wins_mario,
            "draws": draws,
            "s_win_rate": wins_s / total,
            "mario_win_rate": wins_mario / total,
            "draw_rate": draws / total,
            "mean_moves": group["num_moves"].mean(),
            "mean_total_game_time": group["total_game_time"].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "s_vs_mario_50_summary.csv", index=False)

# Grafica 12: resultados globales y separados por quien inicia.
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

overall = summary[summary["starter"] == "Todos"].iloc[0]
labels = ["S heuristico", "Mario", "Empate"]
values = [overall["s_wins"], overall["mario_wins"], overall["draws"]]
colors = [COLORS[S_NAME], COLORS[MARIO_NAME], COLORS["Draw"]]
axes[0].bar(labels, values, color=colors, width=0.65)
axes[0].set_title("Resultado global en 50 partidas", fontweight="bold")
axes[0].set_ylabel("Partidas")
axes[0].set_ylim(0, max(values) + 5)
for x, value in enumerate(values):
    axes[0].text(
        x,
        value + 0.8,
        f"{int(value)}\n{pct(value / overall['games'])}",
        ha="center",
        fontweight="bold",
    )
setup(axes[0])

by_starter = summary[summary["starter"] != "Todos"].copy()
by_starter["starter_label"] = by_starter["starter"].replace(
    {
        S_NAME: "Inicia S",
        MARIO_NAME: "Inicia Mario",
    }
)
x = np.arange(len(by_starter))
bottom = np.zeros(len(by_starter))
for col, label, color in [
    ("s_win_rate", "Gana S", COLORS[S_NAME]),
    ("mario_win_rate", "Gana Mario", COLORS[MARIO_NAME]),
    ("draw_rate", "Empate", COLORS["Draw"]),
]:
    vals = by_starter[col].to_numpy()
    axes[1].bar(x, vals, bottom=bottom, color=color, label=label, width=0.65)
    for xi, base, value in zip(x, bottom, vals):
        if value >= 0.08:
            axes[1].text(
                xi,
                base + value / 2,
                pct(value),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    bottom += vals
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [
        f"{row.starter_label}\nn={int(row.games)}"
        for row in by_starter.itertuples(index=False)
    ]
)
axes[1].set_ylim(0, 1)
axes[1].set_title("Resultado segun quien inicia", fontweight="bold")
axes[1].set_ylabel("Proporcion de partidas")
axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
setup(axes[1])

fig.suptitle("Grafica 12. S vs Mario: resultados alternando inicio", fontsize=15, fontweight="bold")
save(fig, "12_s_vs_mario_resultados.png")

# Grafica 13: costo computacional y duracion.
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

time_df = df.copy()
data = [
    time_df[time_df["focal_agent"] == S_NAME]["mean_decision_time"].to_numpy(),
    time_df[time_df["focal_agent"] == MARIO_NAME]["mean_decision_time"].to_numpy(),
]
box = axes[0].boxplot(
    data, tick_labels=["S heuristico", "Mario"], patch_artist=True
)
for patch, color in zip(box["boxes"], [COLORS[S_NAME], COLORS[MARIO_NAME]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[0].set_title("Tiempo promedio por jugada", fontweight="bold")
axes[0].set_ylabel("Segundos")
setup(axes[0])

axes[1].plot(
    games["game_id"] + 1,
    games["total_game_time"],
    color="#264653",
    linewidth=2,
    label="Tiempo total",
)
axes[1].scatter(games["game_id"] + 1, games["total_game_time"], color="#264653", s=30)
axes_moves = axes[1].twinx()
axes_moves.plot(
    games["game_id"] + 1,
    games["num_moves"],
    color="#d95f02",
    linewidth=2,
    alpha=0.75,
    label="Movimientos",
)
axes[1].set_title("Duracion por partida", fontweight="bold")
axes[1].set_xlabel("Partida")
axes[1].set_ylabel("Tiempo total (s)")
axes_moves.set_ylabel("Movimientos")
lines, labels = axes[1].get_legend_handles_labels()
lines2, labels2 = axes_moves.get_legend_handles_labels()
axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
setup(axes[1])

fig.suptitle("Grafica 13. S vs Mario: costo computacional", fontsize=15, fontweight="bold")
save(fig, "13_s_vs_mario_tiempos.png")

print(f"Graficas guardadas en: {OUT_DIR}")
print(summary.to_string(index=False))
