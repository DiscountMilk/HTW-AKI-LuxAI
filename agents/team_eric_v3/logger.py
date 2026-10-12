"""
Logger - liest am Spielende ab, was World, ActionMask und Brain mitgezählt haben.

  log_game(agent)   Konsole: ein Block pro Spiel  +  eine Zeile an training_log.csv
  plot()            Entwicklung über alle Spiele als Grafik (PNG neben der CSV)
  check_map()       Vergleicht World mit der echten Karte des Spiels

Eigenständig:
  .venv/bin/python -m agents.team_eric_v3.logger          -> plot()
  .venv/bin/python -m agents.team_eric_v3.logger map      -> check_map()

Neue Kennzahl: in collect() eine Zeile ergänzen. Neue Regel / neuer Reward-Term / neue Aktion
erscheinen von selbst (über ActionMask.counts, Reward.totals, Brain.picks).
"""
import csv
import os
from datetime import datetime

import numpy as np

LOG_PATH = os.path.join(os.path.dirname(__file__), "training_log.csv")
ACTION_NAMES = ["stay", "up", "right", "down", "left"]   # Index = Lux-Aktion
WINDOW = 10                                               # Spiele pro gleitendem Schnitt im Plot


def pct(part, whole):
    return part / whole if whole else 0.0


def collect(agent):
    """Alle Kennzahlen eines Spiels als flaches dict (= eine CSV-Zeile)."""
    w, mask, brain = agent.world, agent.action_mask, agent.brain
    rewards, forbids = brain.rewards, sum(mask.counts.values())
    reward_sum = sum(rewards.totals.values())
    return {
        # Ergebnis
        "siege": w.team_wins,
        "niederlagen": w.opp_wins,
        "punkte": w.total_points,
        "relics": len(w.relics),
        "punktefelder": int((w.point_tile == 1).sum()),
        # Action Mask: wie oft greift überhaupt eine Regel, und welche wie stark
        "maske_quote": pct(mask.hits, mask.checks),
        **{f"regel_{name}": pct(n, forbids) for name, n in mask.counts.items()},
        # Aktionen des Netzes
        **{f"aktion_{name}": pct(brain.picks[a], brain.picks.sum()) for a, name in enumerate(ACTION_NAMES)},
        # Reward: Ø pro Netz-Entscheidung und Anteil je Term
        "reward": pct(reward_sum, rewards.count),
        **{f"reward_{name}": pct(v, reward_sum) for name, v in rewards.totals.items()},
        # Netz: Loss = Lernfehler, Q = erwartete Summe künftiger Rewards (wächst beim Lernen, dann stabil)
        "loss": float(np.mean(brain.losses)) if brain.losses else 0.0,
        "q": float(np.mean(brain.qs)) if brain.qs else 0.0,
        "lernschritte": brain.train_steps,
        "epsilon": brain.epsilon,
        "pth_geladen": int(brain.loaded),
    }


class Logger:
    def __init__(self, path=LOG_PATH):
        self.path = path

    # ------------------------------------------------------------------
    # Pro Spiel: Konsole + CSV
    # ------------------------------------------------------------------
    def log_game(self, agent):
        row = collect(agent)
        game = self._append(row)
        self._print(game, row)

    def _append(self, row):
        """Zeile anhängen; ändern sich die Spalten, wird die alte CSV umbenannt. -> Nummer des Spiels"""
        rows = self.read()
        if rows and list(rows[0]) != ["spiel", *row]:
            os.replace(self.path, self.path.replace(".csv", f"_{datetime.now():%Y%m%d_%H%M%S}.csv"))
            rows = []
        game = len(rows) + 1
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["spiel", *row])
            if not rows:
                writer.writeheader()
            writer.writerow({"spiel": game, **{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}})
        return game

    def _print(self, game, r):
        result = "Sieg" if r["siege"] > r["niederlagen"] else "Niederlage"
        part = lambda prefix: " · ".join(f"{k[len(prefix):]} {v:.0%}" for k, v in r.items() if k.startswith(prefix))
        line = "═" * 72
        print(f"\n{line}\n Spiel {game}   ·   {'Start mit agent.pth' if r['pth_geladen'] else 'Start ohne agent.pth (neues Netz)'}"
              f"   ·   {r['lernschritte']:,} Lernschritte".replace(",", "."))
        print(line)
        print(f" ERGEBNIS   {result} {r['siege']}:{r['niederlagen']}   ·   {r['punkte']} Punkte"
              f"   ·   Relics {r['relics']}/6   ·   Punktefelder {r['punktefelder']}")
        print(f" MASKE      greift bei {r['maske_quote']:.0%} der Entscheidungen   →   {part('regel_')}")
        print(f" AKTIONEN   {part('aktion_')}")
        print(f" REWARD     Ø {r['reward']:.3f} pro Entscheidung   →   {part('reward_')}")
        print(f" NETZ       Loss {r['loss']:.3f}   ·   Ø Q {r['q']:.2f}   ·   ε {r['epsilon']:.2f}")

    # ------------------------------------------------------------------
    # Entwicklung über alle Spiele
    # ------------------------------------------------------------------
    def read(self):
        if not os.path.isfile(self.path):
            return []
        with open(self.path, newline="") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]

    def plot(self, show=True):
        """Dünn = einzelne Spiele, dick = Schnitt über WINDOW Spiele. Speichert PNG neben der CSV."""
        import matplotlib.pyplot as plt        # erst hier: der Agent selbst braucht matplotlib nicht

        rows = self.read()
        if not rows:
            print(f"Keine Daten in {self.path}")
            return
        cols = list(rows[0])
        panels = [                             # Titel -> Spalten
            ("Punkte pro Spiel ↑", ["punkte"]),
            ("Matches gewonnen ↑", ["siege"]),
            ("Erkundung ↑", ["relics", "punktefelder"]),
            ("Reward pro Entscheidung ↑", ["reward"]),
            ("Action Mask: greift / Anteil je Regel", ["maske_quote", *[c for c in cols if c.startswith("regel_")]]),
            ("Aktionen des Netzes", [c for c in cols if c.startswith("aktion_")]),
            ("Loss (stabil = gut)", ["loss"]),
            ("Ø Q (steigt, dann stabil)", ["q"]),
        ]
        fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
        games = [r["spiel"] for r in rows]
        for ax, (title, names) in zip(axes.flat, panels):
            for name in names:
                values = np.array([r[name] for r in rows])
                smooth = np.convolve(values, np.ones(WINDOW), "valid") / WINDOW if len(values) >= WINDOW else values
                line, = ax.plot(games[len(games) - len(smooth):], smooth, linewidth=2, label=name)
                if len(names) == 1:
                    ax.plot(games, values, color=line.get_color(), alpha=0.25, linewidth=0.8)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            if len(names) > 1:
                ax.legend(fontsize=7, loc="upper left")
        for ax in axes[-1]:
            ax.set_xlabel("Spiel")
        fig.suptitle(f"{os.path.basename(self.path)} · {len(rows)} Spiele · dick = Ø{WINDOW}")
        fig.tight_layout()
        png = os.path.splitext(self.path)[0] + ".png"
        fig.savefig(png, dpi=110)
        print(f"Plot gespeichert: {png}")
        if show:
            plt.show()


# ----------------------------------------------------------------------
# World gegen die echte Karte prüfen (braucht den verdeckten Spielzustand -> nur eigenständig)
# ----------------------------------------------------------------------
def check_map(games=1, seed=0):
    """Spielt v3 gegen v3 (Evaluation) und vergleicht am Ende jedes Matches World mit der Wahrheit."""
    from luxai_s3.params import EnvParams, env_params_ranges
    from luxai_s3.wrappers import LuxAIS3GymEnv

    from agents.team_eric_v3.agent import Agent
    from agents.team_eric_v3.world import MAX_STEPS_IN_MATCH

    pth = os.path.join(os.path.dirname(__file__), "agent.pth")
    env, rng = LuxAIS3GymEnv(numpy_output=True), np.random.default_rng(seed)
    print(f"\n{'Spiel·Match':<12}{'Spieler':<10}{'Felder bekannt':>15}{'davon richtig':>15}"
          f"{'frisch richtig':>16}{'Relics':>9}{'Punktefelder':>14}{'falsch':>8}")
    for g in range(games):
        agents = [Agent("player_0", agent_path=pth), Agent("player_1", agent_path=pth)]
        params = EnvParams(**{k: rng.choice(v).item() for k, v in env_params_ranges.items()}, match_count_per_episode=3)
        obs, info = env.reset(seed=int(rng.integers(1_000_000)), options={"params": params})
        for a in agents:
            a.set_to_eval_mode()
            a.set_env_cfg(info["params"])
        done, step = False, 0
        while not done:
            obs, _, term, trunc, info = env.step({a.player: a.act(step, obs[a.player]) for a in agents})
            done, step = bool(term["player_0"] or trunc["player_0"]), step + 1
            st = info.get("final_state", info.get("state"))
            for a in agents:
                a.world.update(obs[a.player])
                w = a.world
                if w.match_steps != MAX_STEPS_IN_MATCH:
                    continue
                true_tiles = np.array(st.map_features.tile_type)
                weights = np.array(st.relic_nodes_map_weights)
                active = (weights > 0) & (weights <= int(np.array(st.relic_nodes_mask).sum()) // 2)
                known, fresh = w.last_seen >= 0, w.last_seen >= w.steps - 10
                print(f"{g + 1}·{w.steps // (MAX_STEPS_IN_MATCH + 1) + 1:<10}{a.player:<10}"
                      f"{known.mean():>15.0%}{(w.tile_type == true_tiles)[known].mean():>15.0%}"
                      f"{(w.tile_type == true_tiles)[fresh].mean():>16.0%}"
                      f"{f'{len(w.relics)}/{int(np.array(st.relic_nodes_mask).sum())}':>9}"
                      f"{f'{int(((w.point_tile == 1) & active).sum())}/{int(active.sum())}':>14}"
                      f"{int(((w.point_tile == 1) & ~active).sum()):>8}")
    print("\nFelder bekannt: je gesehen (direkt oder gespiegelt) · davon richtig: Feldtyp stimmt heute noch"
          "\n(Asteroiden/Nebel driften) · frisch: in den letzten 10 Schritten gesehen · falsch: Punktefeld, das keins ist")


if __name__ == "__main__":
    import sys

    check_map() if sys.argv[1:] == ["map"] else Logger().plot()
