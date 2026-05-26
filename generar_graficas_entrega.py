from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "resultados_experimentos.csv"
OUT_DIR = ROOT / "figuras_entrega"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
for col in ["trials", "opponent_trials", "configuration_trials"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

summary = (
    df.groupby(["focal_agent", "opponent", "focal_color", "trials"], dropna=False)
    .agg(
        games=("result", "count"),
        wins=("result", lambda x: (x == "win").sum()),
        draws=("result", lambda x: (x == "draw").sum()),
        losses=("result", lambda x: (x == "loss").sum()),
        win_rate=("result", lambda x: (x == "win").mean()),
        draw_rate=("result", lambda x: (x == "draw").mean()),
        loss_rate=("result", lambda x: (x == "loss").mean()),
        mean_score=("score", "mean"),
        mean_moves=("num_moves", "mean"),
        mean_decision_time=("mean_decision_time", "mean"),
        mean_game_time=("total_game_time", "mean"),
    )
    .reset_index()
)
summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

COLORS = {
    "green": "#1b9e77",
    "yellow": "#d9a441",
    "red": "#d95f02",
    "blue": "#277da1",
    "navy": "#264653",
}
LABELS = {
    "Agente con rollouts heuristicos": "Rollouts heuristicos",
    "Agente con rollouts aleatorios": "Rollouts aleatorios",
}

AGENTE_HEURISTICO = "Agente con rollouts heuristicos"
AGENTE_ALEATORIO = "Agente con rollouts aleatorios"


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def setup(ax):
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, filename: str):
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def trial_labels(data):
    return [f"{int(t)}\nn={int(n)}" for t, n in zip(data["trials"], data["games"])]


random_agg = (
    summary[
        (summary.focal_agent.isin([AGENTE_ALEATORIO, AGENTE_HEURISTICO]))
        & (summary.opponent == "Random")
    ]
    .groupby(["focal_agent", "trials"])
    .agg(
        games=("games", "sum"),
        wins=("wins", "sum"),
        draws=("draws", "sum"),
        losses=("losses", "sum"),
        win_rate=("win_rate", "mean"),
        draw_rate=("draw_rate", "mean"),
        loss_rate=("loss_rate", "mean"),
        mean_decision_time=("mean_decision_time", "mean"),
    )
    .reset_index()
)
random_agg.to_csv(OUT_DIR / "random_matchups_summary.csv", index=False)
all_trials = sorted(random_agg.trials.dropna().astype(int).unique())

# 1. Win / Draw / Loss contra Random.
fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharey=True)
panels = [
    (AGENTE_ALEATORIO, "first"),
    (AGENTE_ALEATORIO, "second"),
    (AGENTE_HEURISTICO, "first"),
    (AGENTE_HEURISTICO, "second"),
]
for ax, (agent, color) in zip(axes.ravel(), panels):
    data = summary[
        (summary.focal_agent == agent)
        & (summary.opponent == "Random")
        & (summary.focal_color == color)
    ].sort_values("trials")
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for metric, label, c in [
        ("win_rate", "Victoria", COLORS["green"]),
        ("draw_rate", "Empate", COLORS["yellow"]),
        ("loss_rate", "Derrota", COLORS["red"]),
    ]:
        vals = data[metric].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=c, label=label, width=0.68)
        for xi, b, v in zip(x, bottom, vals):
            if v >= 0.08:
                ax.text(
                    xi,
                    b + v / 2,
                    pct(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels(data), fontsize=10)
    ax.set_title(
        f"{LABELS[agent]} vs Random\n"
        f"juega {'primero' if color == 'first' else 'segundo'}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Trials por accion\n(n = partidas)")
    ax.set_ylim(0, 1)
    setup(ax)
axes[0, 0].set_ylabel("Proporcion de partidas")
axes[1, 0].set_ylabel("Proporcion de partidas")
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
)
fig.suptitle(
    "Grafica 1. Resultados contra Random",
    fontsize=16,
    fontweight="bold",
    y=0.985,
)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
save(fig, "01_wdl_vs_trials_random.png")


def plot_random_rate(metric, title, ylabel, filename, y_max=1.1):
    fig, ax = plt.subplots(figsize=(12, 7))
    for agent, marker, color in [
        (AGENTE_ALEATORIO, "o", COLORS["blue"]),
        (AGENTE_HEURISTICO, "s", COLORS["green"]),
    ]:
        data = random_agg[random_agg.focal_agent == agent].sort_values("trials")
        xs = [all_trials.index(int(t)) for t in data.trials]
        ax.plot(
            xs,
            data[metric],
            marker=marker,
            linewidth=2.5,
            markersize=8,
            color=color,
            label=LABELS[agent],
        )
        for x, y in zip(xs, data[metric]):
            ax.text(
                x,
                min(y + 0.045, y_max - 0.02),
                pct(y),
                ha="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
    ax.set_xticks(range(len(all_trials)))
    ax.set_xticklabels([str(t) for t in all_trials], fontsize=11)
    ax.set_xlabel("Trials por accion")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, y_max)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    setup(ax)
    save(fig, filename)


plot_random_rate(
    "win_rate",
    "Grafica 2. Win rate vs trials contra Random",
    "Win rate contra Random",
    "02_win_rate_vs_trials_random.png",
)
plot_random_rate(
    "loss_rate",
    "Grafica 3. Tasa de derrotas contra Random",
    "Loss rate contra Random",
    "03_loss_rate_vs_trials_random.png",
    y_max=0.45,
)

# 4 y 9. Comparacion directa entre versiones.
hvr = summary[
    (summary.focal_agent == AGENTE_HEURISTICO)
    & (summary.opponent == AGENTE_ALEATORIO)
]
hvr_agg = (
    hvr.groupby("trials")
    .agg(
        games=("games", "sum"),
        wins=("wins", "sum"),
        draws=("draws", "sum"),
        losses=("losses", "sum"),
        win_rate=("win_rate", "mean"),
        draw_rate=("draw_rate", "mean"),
        loss_rate=("loss_rate", "mean"),
        mean_score=("mean_score", "mean"),
        mean_decision_time=("mean_decision_time", "mean"),
    )
    .reset_index()
)
hvr.to_csv(OUT_DIR / "head_to_head_h_vs_r_by_color_summary.csv", index=False)
hvr_agg.to_csv(OUT_DIR / "head_to_head_h_vs_r_summary.csv", index=False)
h_trials = sorted(hvr.trials.dropna().astype(int).unique())

fig, ax = plt.subplots(figsize=(12, 7))
for color_name, linestyle, marker, c in [
    ("first", "-", "o", COLORS["green"]),
    ("second", "--", "D", COLORS["navy"]),
]:
    data = hvr[hvr.focal_color == color_name].sort_values("trials")
    xs = [h_trials.index(int(t)) for t in data.trials]
    ax.plot(
        xs,
        data.win_rate,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.5,
        markersize=8,
        color=c,
        label=f"Heuristico juega {'primero' if color_name == 'first' else 'segundo'}",
    )
    for x, y in zip(xs, data.win_rate):
        ax.text(x, min(y + 0.045, 1.05), pct(y), ha="center", fontsize=9, color=c)
ax.set_xticks(range(len(h_trials)))
ax.set_xticklabels([str(t) for t in h_trials], fontsize=11)
ax.set_xlabel("Trials por accion en ambos agentes")
ax.set_ylabel("Win rate del agente heuristico")
ax.set_ylim(-0.05, 1.1)
ax.set_title("Grafica 4. Comparacion directa entre versiones", fontsize=15, fontweight="bold")
ax.legend(frameon=False)
setup(ax)
save(fig, "04_head_to_head_h_vs_r.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
data = hvr_agg.sort_values("trials")
x = np.arange(len(data))
bottom = np.zeros(len(data))
for metric, label, c in [
    ("win_rate", "Victoria H", COLORS["green"]),
    ("draw_rate", "Empate", COLORS["yellow"]),
    ("loss_rate", "Derrota H", COLORS["red"]),
]:
    vals = data[metric].to_numpy()
    axes[0].bar(x, vals, bottom=bottom, color=c, label=label, width=0.68)
    for xi, b, v in zip(x, bottom, vals):
        if v >= 0.08:
            axes[0].text(xi, b + v / 2, pct(v), ha="center", va="center", color="white")
    bottom += vals
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{int(t)}\nn={int(n)}" for t, n in zip(data.trials, data.games)])
axes[0].set_ylim(0, 1)
axes[0].set_title("Resultado agregado del agente heuristico")
axes[0].set_xlabel("Trials en ambos agentes")
axes[0].set_ylabel("Proporcion")
axes[0].legend(frameon=False)
setup(axes[0])
axes[1].plot(x, data.mean_score, marker="o", linewidth=2.5, color=COLORS["green"], label="Score heuristico")
axes[1].set_ylim(-1.05, 1.05)
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(int(t)) for t in data.trials])
axes[1].set_xlabel("Trials en ambos agentes")
axes[1].set_ylabel("Score medio")
axes2 = axes[1].twinx()
axes2.plot(x, data.mean_decision_time, marker="s", linewidth=2.5, color=COLORS["navy"], label="Tiempo heuristico")
axes2.set_ylabel("Tiempo medio de decision heuristica (s)")
for xi, score, secs in zip(x, data.mean_score, data.mean_decision_time):
    axes[1].text(xi, score + 0.08 if score < 0.9 else score - 0.15, f"{score:.2f}", ha="center")
    axes2.text(xi, secs + 0.05, f"{secs:.2f}s", ha="center", fontsize=9)
lines, labels = axes[1].get_legend_handles_labels()
lines2, labels2 = axes2.get_legend_handles_labels()
axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")
axes[1].set_title("Score y costo del agente heuristico")
setup(axes[1])
fig.suptitle(
    "Grafica 9. Metricas: heuristico vs rollouts aleatorios",
    fontsize=15,
    fontweight="bold",
)
save(fig, "09_metricas_heuristico_vs_aleatorio.png")

# 5. Self-play.
self_df = df[(df.agent_1 == df.agent_2) & (df.focal_side == "agent_1")]
self_summary = (
    self_df.groupby(["agent_1", "configuration_trials"])
    .agg(
        games=("result", "count"),
        mean_score=("score", "mean"),
        draw_rate=("result", lambda x: (x == "draw").mean()),
        mean_moves=("num_moves", "mean"),
    )
    .reset_index()
)
self_summary.to_csv(OUT_DIR / "self_play_summary.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for agent, marker, c in [
    (AGENTE_ALEATORIO, "o", COLORS["blue"]),
    (AGENTE_HEURISTICO, "s", COLORS["green"]),
]:
    data = self_summary[self_summary.agent_1 == agent].sort_values("configuration_trials")
    xs = np.arange(len(data))
    labels = [str(int(t)) for t in data.configuration_trials]
    axes[0].plot(xs, data.mean_score, marker=marker, linewidth=2.5, color=c, label=LABELS[agent])
    axes[1].plot(xs, data.draw_rate, marker=marker, linewidth=2.5, color=c, label=LABELS[agent])
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
axes[0].axhline(0, color="black", linewidth=1, alpha=0.5)
axes[0].set_title("Score promedio del jugador 1")
axes[0].set_ylabel("Score promedio")
axes[1].set_title("Draw rate en self-play")
axes[1].set_ylabel("Draw rate")
for ax in axes:
    ax.set_xlabel("Trials por accion")
    ax.legend(frameon=False)
    setup(ax)
fig.suptitle("Grafica 5. Autojuego / self-play", fontsize=15, fontweight="bold")
save(fig, "05_self_play_score.png")

# 6. Heatmap.
heat_df = df[(df.configuration_trials == 50) & (df.focal_side == "agent_1")]
pivot = heat_df.groupby(["agent_1", "agent_2"])["score"].mean().unstack()
counts = heat_df.groupby(["agent_1", "agent_2"])["score"].count().unstack()
pivot.to_csv(OUT_DIR / "heatmap_score_matrix.csv")
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(pivot.fillna(0), vmin=-1, vmax=1, cmap="RdYlGn")
ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_xlabel("Jugador 2")
ax.set_ylabel("Jugador 1")
ax.set_title("Grafica 6. Score promedio del jugador 1 (trials=50)", fontsize=15, fontweight="bold")
for i, row in enumerate(pivot.index):
    for j, col in enumerate(pivot.columns):
        value = pivot.loc[row, col]
        if pd.notna(value):
            ax.text(j, i, f"{value:.2f}\nn={int(counts.loc[row, col])}", ha="center", va="center", fontweight="bold")
fig.colorbar(im, ax=ax, label="Score promedio")
save(fig, "06_heatmap_score_trials_50.png")

# 7. Time per move.
fig, ax = plt.subplots(figsize=(12, 7))
for agent, marker, c in [
    (AGENTE_ALEATORIO, "o", COLORS["blue"]),
    (AGENTE_HEURISTICO, "s", COLORS["green"]),
]:
    data = random_agg[random_agg.focal_agent == agent].sort_values("trials")
    xs = [all_trials.index(int(t)) for t in data.trials]
    ax.plot(xs, data.mean_decision_time, marker=marker, linewidth=2.5, markersize=8, color=c, label=LABELS[agent])
    for x, y in zip(xs, data.mean_decision_time):
        ax.text(x, y * 1.05 + 0.02, f"{y:.2f}s", ha="center", fontsize=9, color=c)
ax.set_xticks(range(len(all_trials)))
ax.set_xticklabels([str(t) for t in all_trials])
ax.set_xlabel("Trials por accion")
ax.set_ylabel("Tiempo promedio por jugada (s)")
ax.set_title("Grafica 7. Costo computacional vs trials contra Random", fontsize=15, fontweight="bold")
ax.legend(frameon=False)
setup(ax)
save(fig, "07_time_per_move_vs_trials.png")

# 8. Performance vs cost.
fig, ax = plt.subplots(figsize=(12, 7))
for agent, marker, c in [
    (AGENTE_ALEATORIO, "o", COLORS["blue"]),
    (AGENTE_HEURISTICO, "s", COLORS["green"]),
]:
    data = random_agg[random_agg.focal_agent == agent].sort_values("trials")
    ax.plot(data.mean_decision_time, data.win_rate, alpha=0.35, color=c)
    ax.scatter(data.mean_decision_time, data.win_rate, marker=marker, s=95, color=c, label=LABELS[agent])
    for _, row in data.iterrows():
        ax.annotate(f"{int(row.trials)} trials", (row.mean_decision_time, row.win_rate), textcoords="offset points", xytext=(7, 7), fontsize=9)
ax.set_xlabel("Tiempo promedio por jugada (s)")
ax.set_ylabel("Win rate contra Random")
ax.set_ylim(-0.05, 1.08)
ax.set_title("Grafica 8. Rendimiento vs costo computacional", fontsize=15, fontweight="bold")
ax.legend(frameon=False)
setup(ax)
save(fig, "08_performance_vs_cost.png")

print(f"Graficas guardadas en: {OUT_DIR}")
