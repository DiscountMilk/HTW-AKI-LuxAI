"""
logger.py = Kennzahlen pro Spiel: Block in der Konsole + eine Zeile in logging/training_log.csv.
Zählt nur mit, was agent.py ihm gibt. Entscheidet nichts.

    on_act(actions, alive, mask)   jeden Zug: gewählte Aktionen und Maske zählen
    on_update(world, alive)        jeden Trainingszug: Punkte und Reward-Terme zählen
    log_game(player, world, model) Spielende: CSV-Zeile + Konsole
    eval_step(player, world, last) nur bei train=False, jeden Zug: mitzählen, im letzten Zug eval_log.csv schreiben
    plot()                         Entwicklung über alle Spiele (PNG neben der CSV)

Ein Logger pro Spiel und Spieler (agent.py legt ihn in set_env_cfg neu an).
Selbstspiel: zwei Zeilen pro Spiel (Spalte "spieler"), Netz-Werte sind bei beiden gleich (ein Modell).
Spalte "gegner": Name (agent.name) des anderen Agents, aus dem Runner gelesen.

Eigenständig:  .venv/bin/python -m agents.team_eric_v5.logger   -> plot()
Neue Kennzahl: eine Zeile in row() ergänzen. Neue Reward-Terme erscheinen von selbst.
"""
import csv
import inspect
import os
from datetime import datetime

import numpy as np

from agents.team_eric_v5.reward import TERMS, reward_parts

LOG_PATH = os.path.join(os.path.dirname(__file__), "logging", "training_log.csv")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "logging", "eval_log.csv")
TRAIN_ONLY = ("loss", "q", "lernschritte", "trainingszuege", "epsilon")   # fehlen in eval_log.csv
ACTION_NAMES = ["stay", "up", "right", "down", "left", "sap"]   # Index = Aktions-Typ
WINDOW = 10                                               # Zeilen pro gleitendem Schnitt im Plot


class Logger:
    def __init__(self, model, path=LOG_PATH):
        self.path = path
        self.decisions = self.masked = self.only_stay = self.points = 0
        self.picks = np.zeros(len(ACTION_NAMES), dtype=int)
        self.rewards = {f.__name__: 0.0 for f, _ in TERMS}
        self.start = (model.updates, model.loss_sum, model.q_sum)    # Netz-Werte nur für dieses Spiel

    # ------------------------------------------------------------------
    # Mitzählen
    # ------------------------------------------------------------------
    def on_act(self, actions, alive, mask):
        units = np.flatnonzero(alive)
        m = np.asarray(mask, dtype=bool)[units]
        self.decisions += len(units)
        self.masked += int((~m).any(axis=1).sum())                  # mind. eine Aktion verboten
        self.only_stay += int((~m[:, 1:]).all(axis=1).sum())        # keine Bewegung erlaubt
        self.picks += np.bincount(actions[units, 0], minlength=len(ACTION_NAMES))

    def on_update(self, w, alive):
        """Nach World.update(next_obs). alive = Units, die im letzten Zug gehandelt haben."""
        self.points += int(w.points_gained)
        for name, r in reward_parts(w).items():
            self.rewards[name] += float(r[alive].sum())

    # ------------------------------------------------------------------
    # Spielende
    # ------------------------------------------------------------------
    def row(self, w, model):
        """Alle Kennzahlen eines Spiels als flaches dict (= eine CSV-Zeile)."""
        d = max(self.decisions, 1)
        n = model.updates - self.start[0] if model else 0
        return {
            # Ergebnis
            "siege": w.my_wins,
            "niederlagen": w.opp_wins,
            "punkte": self.points,
            "relics": len(w.relic_positions),
            "punktefelder": int((w.point_tile == 1).sum()),
            "karte_bekannt": float((w.last_seen >= 0).mean()),
            # Maske: Anteil der Entscheidungen
            "maske_quote": self.masked / d,
            "maske_nur_stay": self.only_stay / d,
            # Aktionen des Netzes
            **{f"aktion_{name}": self.picks[a] / d for a, name in enumerate(ACTION_NAMES)},
            # Reward: Ø pro Entscheidung, gesamt und je Term (mit Gewicht)
            "reward": sum(self.rewards.values()) / d,
            **{f"reward_{name}": v / d for name, v in self.rewards.items()},
            # Netz: Loss = Lernfehler, Q = erwartete Summe künftiger Rewards
            **({"loss": (model.loss_sum - self.start[1]) / n if n else 0.0,
                "q": (model.q_sum - self.start[2]) / n if n else 0.0,
                "lernschritte": model.updates,
                "trainingszuege": model.steps,
                "epsilon": model.epsilon()} if model else {}),
        }

    def log_game(self, player, w, model):
        row = {"spieler": player, "gegner": opponent_name(player), **self.row(w, model)}
        game = self._append(row, self.path)
        self._print(game, row)

    def eval_step(self, player, w, last):
        """train=False: Der Runner ruft kein update() auf und liefert das Spielende nicht.
        Darum hier jeden Zug mitzählen und im letzten Zug des letzten Matches schreiben."""
        if last is not None:
            self.on_update(w, last["alive"])
        if w.match_nr < w.match_count_per_episode - 1 or w.match_step < w.max_steps_in_match:
            return
        row = {"spieler": player, "gegner": opponent_name(player),
               **{k: v for k, v in self.row(w, None).items() if k not in TRAIN_ONLY}}
        # ponytail: letztes Match nach Punktestand vor dem letzten Zug gezählt (Endergebnis sieht der Agent nie)
        row["siege"] += int(w.my_points > w.opp_points)
        row["niederlagen"] += int(w.my_points < w.opp_points)
        game = self._append(row, EVAL_PATH)
        self._print(game, row)

    def _append(self, row, path):
        """Zeile anhängen; ändern sich die Spalten, wird die alte CSV umbenannt. -> Nummer des Spiels"""
        rows = read(path)
        if rows and list(rows[0]) != ["spiel", *row]:
            os.replace(path, path.replace(".csv", f"_{datetime.now():%Y%m%d_%H%M%S}.csv"))
            rows = []
        game = sum(r["spieler"] == row["spieler"] for r in rows) + 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["spiel", *row])
            if not rows:
                writer.writeheader()
            writer.writerow({"spiel": game, **{k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                                               for k, v in row.items()}})
        return game

    def _print(self, game, r):
        result = "Sieg" if r["siege"] > r["niederlagen"] else "Niederlage"
        part = lambda prefix, fmt: " · ".join(f"{k[len(prefix):]} {v:{fmt}}" for k, v in r.items() if k.startswith(prefix))
        line = "═" * 72
        net = f"   ·   {r['lernschritte']:,} Lernschritte   ·   {r['trainingszuege']:,} Trainingszüge" if "loss" in r else "   ·   Evaluation"
        print(f"\n{line}\n Spiel {game} · Spieler {r['spieler']} gegen {r['gegner']}" + net.replace(",", "."))
        print(line)
        print(f" ERGEBNIS   {result} {r['siege']}:{r['niederlagen']}   ·   {r['punkte']} Punkte   ·   Relics {r['relics']}"
              f"   ·   Punktefelder {r['punktefelder']}   ·   Karte {r['karte_bekannt']:.0%} bekannt")
        print(f" MASKE      greift bei {r['maske_quote']:.0%} der Entscheidungen   ·   nur STAY {r['maske_nur_stay']:.0%}")
        print(f" AKTIONEN   {part('aktion_', '.0%')}")
        print(f" REWARD     Ø {r['reward']:.3f} pro Entscheidung   →   {part('reward_', '.3f')}")
        if "loss" in r:
            print(f" NETZ       Loss {r['loss']:.3f}   ·   Ø Q {r['q']:.2f}   ·   ε {r['epsilon']:.2f}")


# ----------------------------------------------------------------------
# Entwicklung über alle Spiele
# ----------------------------------------------------------------------
def read(path=LOG_PATH):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v):
    """Zahl, wenn möglich (Spalte "gegner" bleibt Text)."""
    try:
        return float(v)
    except ValueError:
        return v


def opponent_name(player):
    """Name des Gegners. lux/runner.py hält beide Agents als lokale Variablen player_0 / player_1."""
    # ponytail: liest den Aufruf-Stack des Runners; ändert der Runner diese Namen, steht hier "unbekannt"
    f = inspect.currentframe()
    while f:
        opp = f.f_locals.get(f"player_{1 - player}")
        if opp is not None:
            return getattr(opp, "name", type(opp).__name__)
        f = f.f_back
    return "unbekannt"


def plot(path=LOG_PATH, show=True):
    """Dünn = einzelne Zeilen, dick = Schnitt über WINDOW Zeilen. Speichert PNG neben der CSV."""
    import matplotlib.pyplot as plt        # erst hier: der Agent selbst braucht matplotlib nicht

    rows = read(path)
    if not rows:
        print(f"Keine Daten in {path}")
        return
    cols = list(rows[0])
    panels = [                             # Titel -> Spalten
        ("Punkte pro Spiel ↑", ["punkte"]),
        ("Matches gewonnen ↑", ["siege"]),
        ("Erkundung ↑", ["relics", "punktefelder"]),
        ("Karte bekannt ↑", ["karte_bekannt"]),
        ("Reward pro Entscheidung ↑", ["reward"]),
        ("Reward je Term", [c for c in cols if c.startswith("reward_")]),
        ("Action Mask", ["maske_quote", "maske_nur_stay"]),
        ("Aktionen des Netzes", [c for c in cols if c.startswith("aktion_")]),
        ("Loss (stabil = gut)", ["loss"]),
        ("Ø Q (steigt, dann stabil)", ["q"]),
    ]
    fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(13, 13), sharex=True)
    x = np.arange(1, len(rows) + 1)
    for ax, (title, names) in zip(axes.flat, panels):
        for name in names:
            values = np.array([r[name] for r in rows])
            smooth = np.convolve(values, np.ones(WINDOW), "valid") / WINDOW if len(values) >= WINDOW else values
            line, = ax.plot(x[len(x) - len(smooth):], smooth, linewidth=2, label=name)
            if len(names) == 1:
                ax.plot(x, values, color=line.get_color(), alpha=0.25, linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if len(names) > 1:
            ax.legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Zeile (1 pro Spiel und Spieler)")
    fig.suptitle(f"{os.path.basename(path)} · {len(rows)} Zeilen · dick = Ø{WINDOW}")
    fig.tight_layout()
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, dpi=110)
    print(f"Plot gespeichert: {png}")
    if show:
        plt.show()


if __name__ == "__main__":
    plot()
