"""
TrainingLog = Protokoll des Trainings. Nach jedem Spiel (nur bei train=True):
  1. eine Zeile an training_log.csv anhängen   (für Kurven, bleibt über alle Läufe erhalten)
  2. Konsole: Werte dieses Spiels + Schnitt der letzten WINDOW Spiele mit Trend zum Schnitt davor
  3. plot(): Entwicklung über alle Spiele als Grafik (Abschnitte wie in der Konsole, PNG neben der CSV)
     .venv/bin/python agents/team_eric_v2/training_log.py [pfad/zur.csv]

Neue Kennzahl: in Agent._log_game() eintragen und in _print() im passenden Abschnitt anzeigen.
Soll sie einen Trend bekommen -> Anzeigename in TREND, dann self._trends(history, "<kennzahl>").
Regeln (safety_*, mask_*) und Reward-Terme (r_*) erscheinen automatisch, ohne Änderung hier.
Ziel jeder Kennzahl (↑ / ↓ / =) steht in GOAL; danach bewertet der Trend "besser", "schlechter", "stabil".
Ändern sich die Kennzahlen, wird die alte CSV umbenannt und eine neue begonnen.
"""
import csv
import os
from datetime import datetime

import numpy as np

ACTION_NAMES = ["stay", "up", "right", "down", "left"]   # Index = Lux-Aktion
WINDOW = 10                                               # Spiele pro gleitendem Schnitt
TREND = {                                                 # Kennzahl -> Name in der Konsole
    "gewonnen": "Siegquote",
    "punkte": "Punkte",
    "reward": "Reward",
    "loss": "Loss",
    "q": "Q",
}
# Was ist gut? "↑" höher besser, "↓" niedriger besser, "=" langfristig stabil besser, fehlt = nur Info
GOAL = {
    "siege": "↑", "gewonnen": "↑", "punkte": "↑", "relics": "↑", "punktefelder": "↑", "reward": "↑",
    "loss": "=",                       # darf schwanken, aber nicht dauerhaft wachsen
    "q": "=",                          # steigt beim Lernen, danach stabil; dauerhaft wachsend = explodiert
    "mask_blocked": "",                # hängt vor allem von der Karte ab
}
GOAL_PREFIX = {"r_": "↑", "mask_": "↓"}   # Standard für automatisch erkannte Spalten
LEGEND = "↑ höher besser | ↓ niedriger besser | = stabil besser"
STABLE = 0.1                              # Änderung bis 10 % gilt bei "=" als stabil
PANELS = [                                # plot(): Titel -> Spalten ("präfix*" = alle passenden Spalten)
    ("Punkte", ["punkte"]),
    ("Siegquote", ["gewonnen"]),
    ("Erkundung", ["relics", "punktefelder"]),
    ("Reward pro Netz-Entscheidung", ["reward", "r_*"]),
    ("Loss", ["loss"]),
    ("Q", ["q"]),
    ("Action Mask (Ø verbotene Aktionen)", ["mask_*"]),
    ("Safety Rules (Anteil Entscheidungen)", ["safety_*"]),
    ("Aktionen des Netzes", ACTION_NAMES),
]


def goal(key):
    """Zielrichtung einer Kennzahl: '↑', '↓', '=' oder '' (nur Info)."""
    if key in GOAL:
        return GOAL[key]
    return next((g for prefix, g in GOAL_PREFIX.items() if key.startswith(prefix)), "")


class TrainingLog:
    def __init__(self, path):
        self.path = path

    def add(self, row):
        """Ein Spiel eintragen und den Fortschritt ausgeben."""
        history = self._read()
        row = {"spiel": len(history) + 1, **row}
        row = {k: v if isinstance(v, int) else round(float(v), 4) for k, v in row.items()}
        if history and list(history[0]) != list(row):   # Kennzahlen geändert -> neue Datei
            os.replace(self.path, self.path.replace(".csv", f"_{datetime.now():%Y%m%d_%H%M%S}.csv"))
            history, row["spiel"] = [], 1
        self._append(row, header=not history)
        self._print(row, history + [row])

    def plot(self, save=True, show=True):
        """Entwicklung über alle Spiele der CSV. Dünn = einzelne Spiele, dick = Schnitt über WINDOW Spiele.
        Panels ohne passende Spalten (z.B. noch keine Safety-Regel) werden weggelassen."""
        import matplotlib.pyplot as plt   # erst hier: der Agent selbst braucht matplotlib nicht

        history = self._read()
        if not history:
            print(f"Keine Daten in {self.path}")
            return
        columns, games = list(history[0]), [h["spiel"] for h in history]
        panels = [(title, [c for p in patterns for c in columns
                           if c == p or (p.endswith("*") and c.startswith(p[:-1]))])
                  for title, patterns in PANELS]
        panels = [(title, cols) for title, cols in panels if cols]

        rows = -(-len(panels) // 2)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 2.8 * rows), sharex=True, squeeze=False)
        for ax, (title, cols) in zip(axes.flat, panels):
            for c in cols:
                values = np.array([h[c] for h in history])
                line, = ax.plot(games, self._rolling(values), linewidth=2, label=f"{c} {goal(c)}".strip())
                if len(cols) == 1:                                     # bei einer Linie auch Einzelspiele zeigen
                    ax.plot(games, values, color=line.get_color(), alpha=0.25, linewidth=0.8)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")
        for ax in axes.flat[len(panels):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Spiel")
        fig.suptitle(f"{os.path.basename(self.path)}: {len(history)} Spiele   (dick = Ø{WINDOW} Spiele | {LEGEND})")
        fig.tight_layout()
        if save:
            png = os.path.splitext(self.path)[0] + ".png"
            fig.savefig(png, dpi=120)
            print(f"Plot gespeichert: {png}")
        if show:
            plt.show()

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------
    def _read(self):
        if not os.path.isfile(self.path):
            return []
        with open(self.path, newline="") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

    def _append(self, row, header):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row)
            if header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _rolling(values):
        """Gleitender Schnitt über die letzten WINDOW Spiele (am Anfang über entsprechend weniger)."""
        cs = np.concatenate([[0.0], np.cumsum(values)])
        end = np.arange(1, len(values) + 1)
        start = np.maximum(end - WINDOW, 0)
        return (cs[end] - cs[start]) / (end - start)

    def _trend(self, history, key):
        """Schnitt der letzten WINDOW Spiele, ab 2*WINDOW Spielen mit Änderung zum Schnitt davor."""
        values = [h[key] for h in history]
        now = np.mean(values[-WINDOW:])
        if len(values) < 2 * WINDOW:
            return f"{now:.2f}"
        before = np.mean(values[-2 * WINDOW:-WINDOW])
        change = now - before
        verdict = self._verdict(goal(key), change, before)
        return f"{now:.2f} ({change:+.2f}{' ' + verdict if verdict else ''})"

    @staticmethod
    def _verdict(g, change, before):
        """Bewertet eine Änderung nach dem Ziel g der Kennzahl."""
        if g == "=":
            if abs(change) <= STABLE * abs(before):
                return "stabil"
            return "steigt" if change > 0 else "sinkt"
        if g in ("↑", "↓"):
            if change == 0:
                return "gleich"
            return "besser" if (change > 0) == (g == "↑") else "schlechter"
        return ""

    def _print(self, row, history):
        """Ein Block pro Spiel: Ergebnis, Safety Rules, Action Mask, Brain, Reward.
        Regeln und Reward-Terme erscheinen automatisch über ihre Spalten-Präfixe safety_, mask_, r_."""
        safety, mask, terms = self._group(row, "safety_"), self._group(row, "mask_"), self._group(row, "r_")
        total = row["reward"]

        print(f"\n{'#' * 30} Spiel {row['spiel']} {'#' * 30}   {LEGEND}")
        self._section("ERGEBNIS",
                      f"Matches{goal('siege')} {row['siege']}:{row['niederlagen']}"
                      f" ({'Sieg' if row['gewonnen'] else 'Niederlage'})"
                      f" | Punkte{goal('punkte')} {row['punkte']} | Relics gefunden{goal('relics')} {row['relics']}"
                      f" | Punktefelder erkannt{goal('punktefelder')} {row['punktefelder']}")
        self._section("SAFETY RULES",
                      f"Regeln {row['regel']:.0%} | Netz {1 - row['regel']:.0%}   (Anteil aller Entscheidungen)",
                      self._join(safety, "{:.0%}", "safety_") or "keine Regel eingetragen")
        self._section("ACTION MASK",
                      self._join(mask, "{:.2f}", "mask_") + "   (Ø verbotene Aktionen pro Netz-Entscheidung)"
                      if mask else "keine Regel eingetragen")
        self._section("BRAIN",
                      f"Loss{goal('loss')} {row['loss']:.3f} | Q{goal('q')} {row['q']:.2f} | Epsilon {row['epsilon']}",
                      "Aktionen " + self._join({a: row[a] for a in ACTION_NAMES}, "{:.0%}"),
                      self._trends(history, "loss", "q"))
        self._section("REWARD",
                      f"Ø{goal('reward')} {total:.3f} pro Netz-Entscheidung = "
                      + " + ".join(f"{k}{goal('r_' + k)} {v:.3f}" + (f" ({v / total:.0%})" if total >= 0.001 else "")
                                   for k, v in terms.items()),
                      self._trends(history, "reward"))

    def _trends(self, history, *keys):
        """'Ø10: Loss= 0.20 (-0.01 stabil) | ...' für die genannten Kennzahlen aus TREND."""
        return f"Ø{len(history[-WINDOW:])}: " + " | ".join(
            f"{TREND[k]}{goal(k)} {self._trend(history, k)}" for k in keys)

    @staticmethod
    def _group(row, prefix):
        """Alle Spalten mit diesem Präfix, Präfix abgeschnitten: {'blocked': 1.06, ...}"""
        return {k[len(prefix):]: v for k, v in row.items() if k.startswith(prefix)}

    @staticmethod
    def _join(values, fmt, prefix=""):
        """'blocked 1.06 | no_energy_to_move↓ 0.44'; prefix + Name ergibt die Spalte für goal()."""
        return " | ".join(f"{k}{goal(prefix + k)} {fmt.format(v)}" for k, v in values.items())

    @staticmethod
    def _section(title, *lines):
        """Titel in der ersten Zeile links, weitere Zeilen bündig darunter."""
        for i, line in enumerate(lines):
            print(f"{title if i == 0 else '':<14}{line}")


if __name__ == "__main__":
    # Entwicklung plotten (Standard: training_log.csv neben dieser Datei):
    #   .venv/bin/python agents/team_eric_v2/training_log.py [pfad/zur.csv]
    import sys

    TrainingLog(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "training_log.csv")).plot()
