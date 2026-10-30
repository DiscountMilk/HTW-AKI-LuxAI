# Team Eric V4 – Architektur

LuxAI Season 3 Agent. Deep-Q-Learning mit einem CNN.
Import-Pfad: `agents.team_eric_v5`

## Grundregeln (KISS)
- Einfach vor clever. Kein Code "für später".
- Eine Datei = eine Aufgabe. Dateien klein halten.
- Kommentare und Namen auf Deutsch, kurz.
- Erst prüfen, ob es etwas schon gibt, dann neu schreiben.

## Ablauf pro Zug
```
obs ──► World.update()   Gedächtnis aktualisieren
        │
        ▼
  FeatureEngineering.encode(world)   Welt → Zahlen  (maps, globals)
        │
        ▼
  Model.choose(x, pos, alive, mask, sap_ok, sap_range)  Aktion pro Unit (Bewegung oder Sap)
        │
        ▼
  actions (max_units, 3)  ──► Spiel

Danach (nur Training):
next_obs ──► World.update() ──► compute_reward(world) ──► Model.train_step(...)
```

## Dateien

| Datei | Aufgabe | Darf NICHT |
|---|---|---|
| `agent.py` | Klebt alles zusammen. Schnittstelle zum Spiel (`act`, `update`, `save/load`). | Logik enthalten |
| `world.py` | Gedächtnis: Karte, Relics, Punkt-Felder, Units, Punkte. | Netz kennen, Entscheidungen treffen |
| `features.py` | World → Zahlen fürs Netz. | Etwas speichern oder entscheiden |
| `action_mask.py` | Welche Aktionen sind für eine Unit erlaubt? | Netz kennen |
| `reward.py` | Belohnung aus der World berechnen. | Netz kennen |
| `model.py` | CNN, Aktionswahl, Training, Replay-Buffer, Speichern. | World direkt lesen (nur Features) |
| `logger.py` | Kennzahlen pro Spiel: Konsole + `logging/training_log.csv`, Plot. | Entscheidungen treffen |
| `mirror.py` | Spieler 1: obs spiegeln, Aktionen zurückdrehen. Selbsttest: `.venv/bin/python -m agents.team_eric_v5.mirror` | Logik außer Spiegeln enthalten |
| `test_sap.py` | Selbsttest Sap: `.venv/bin/python -m agents.team_eric_v5.test_sap` | – |

Status: alle Dateien da, Selbstspiel-Training läuft. In `Paths.csv` als Gruppe 8 (`agents/team_eric_v4/agent.pth`).

## Aktionsmaske (action_mask.py)
- `allowed_actions(world, unit)` → `(5,)` bool, Index = Aktions-Typ 0..4
- Aktion verboten, wenn eine Regel in `RULES` True liefert. Regel = Funktion `(w, unit, action) -> bool`
- Vorerst keine Regel aktiv (alles erlaubt). Vorhanden, aber auskommentiert: `no_energy`, `blocked`
- Verbietet eine Regel alles, erlaubt `model.py` STAY
- `can_sap(world)` → `(N,)` bool: lebt und Energie ≥ `unit_sap_cost`. Nur Engine-Regel, keine Strategie. Reichweite und Kartenrand prüft `model.py` (`_sap_area`)

## Logging (logger.py)
- Ein `Logger` pro Spiel und Spieler (`set_env_cfg`), nur im Training (`update`): zählt in `act`/`update`, schreibt bei `terminated`
- CSV: `logging/training_log.csv`, eine Zeile pro Spiel und Spieler (Selbstspiel: 2 Zeilen, Spalte `spieler`)
- Spalte `gegner`: `name` des anderen Agents, gelesen aus den Runner-Variablen `player_0` / `player_1` (`opponent_name`); nicht gefunden → `unbekannt`
- Spalten ändern sich → alte CSV wird mit Zeitstempel umbenannt
- Evaluation (`train=False`): Runner ruft kein `update()` auf und liefert das Spielende nicht. `act()` ruft `Logger.eval_step` auf; im letzten Zug eine Zeile in `logging/eval_log.csv` (ohne Netz-Spalten). Letztes Match nach Punktestand vor dem letzten Zug gezählt
- Netz-Werte (Loss, Ø Q) aus `Model.loss_sum` / `q_sum` / `updates`, nur über dieses Spiel
- Neue Kennzahl: Zeile in `Logger.row()`. Neue Reward-Terme erscheinen von selbst
- Plot: `.venv/bin/python -m agents.team_eric_v4.logger` → PNG neben der CSV

## Feste Konventionen

**Koordinaten**: immer `array[x, y]`, Form `(W, H)`. Feld (0,0) = Start Spieler 0.

**Aktionen** (LuxAI S3): `[typ, dx, dy]`
| typ | Aktion | Richtung |
|---|---|---|
| 0 | stay | – |
| 1 | up | y − 1 |
| 2 | right | x + 1 |
| 3 | down | y + 1 |
| 4 | left | x − 1 |
| 5 | sap | Ziel (dx, dy) |

**Spiegelung** (Karte ist symmetrisch): `(x, y) → (W−1−y, H−1−x)`
- Funktionen: `World.mirror(array)` und `World.mirror_pos(x, y)`
- Richtungen gespiegelt: up ↔ right, down ↔ left, sap-Ziel `(dx, dy) → (−dy, −dx)`

**Spieler 1**: Das Netz sieht sich immer als Spieler 0.
→ `mirror.py`: `agent.act` spiegelt die obs (`obs_als_spieler0`, Teams getauscht) und dreht die Aktionen zurück (`aktionen_zurueck`). World ist intern immer Team 0.
→ Trainiert wird nur als Spieler 0: `update()` kehrt bei Spieler 1 sofort zurück. Logger bekommt den echten `team_id`.

**Werte im World-Gedächtnis**
- `tile_type`: −1 unbekannt, 0 leer, 1 Nebel, 2 Asteroid
- `point_tile`: −1 unbekannt, 0 kein Punkt, 1 Punkt-Feld
- `−1` bedeutet immer "unbekannt / nie gesehen"

## Netz-Eingabe (features.py)
- `maps`: `(C, W, H)`, aktuell C = 15
- `globals`: `(G,)`, aktuell G = 9
- Werte grob auf 0..1 bzw. −1..1 skalieren
- Kategorien als One-Hot-Kanäle, nicht als eine Zahl
- Neues Feature: Methode schreiben + Zeile in `MAP_PARTS` / `GLOBAL_PARTS`

## Reward (reward.py)
- `compute_reward(world)` → `(max_units,)`, ein Wert pro Unit, Aufruf NACH `World.update(next_obs)`
- Team-Terme (Punkte, Match-Ergebnis, Erkunden, Relic gefunden) bekommt jede Unit gleich
- Unit-Terme: steht auf sicherem Punkt-Feld (geteilt durch eigene Units dort); `stacked` −0.2, wenn die Unit ihr Feld mit eigenen Units teilt; `energy_tile` bis 0.1 je nach positiver Feldenergie (Gewicht muss < `on_point_tile` bleiben, Assert in reward.py); `toward_point_tile` ±0.1 pro Feld näher/weiter zum nächsten freien Ziel (sicheres Punkt-Feld oder ungeprüftes Feld nahe Relic, `world.point_tile_candidates()`) (braucht `world.my_prev_pos` / `my_prev_alive`); `sap_result` 0.2 · (Treffer − 1): sichtbare Gegner nach ihrem Zug, 1 auf dem Zielfeld, 0.5 auf den 8 Nachbarfeldern (braucht `world.merke_saps`)
- Neuer Term: Funktion `(w) -> (max_units,)` + Zeile in `TERMS`
- Was „seit dem letzten Zug“ passiert ist, merkt sich die World:
  `points_gained`, `match_result` (+1/−1 im Zug nach Matchende), `new_tiles`, `new_relics`

## Netz (model.py)
- Vollständig faltend: Ausgang `(Aktionen + 1, W, H)` = Q-Wert pro Bewegung und Feld + Sap-Karte
- Sap-Karte (Kanal `SAP` = 5): Q für „Sap auf dieses Feld“, gelesen am ZIELFELD. Das Netz entscheidet ob und wohin: bestes Feld in Reichweite gegen die gewählte Bewegung
- `globals` als konstante Kanäle an die Karte anhängen
- Q-Werte der Bewegungen = Ausgang an der Position der Unit `(x, y)`
- ε: Sap zählt wie EINE Aktion, dann Zufallsfeld in Reichweite
- `load()` übernimmt alte `.pth` mit 5 Ausgängen: Bewegungs-Kanäle kopiert, Sap-Kanal neu
- Verbotene Aktionen (Maske) → `-inf` vor `argmax` / `max`. Darf eine Unit nichts, ist STAY erlaubt
- Mehrere Units auf einem Feld: k-te Unit nimmt die k-tbeste erlaubte Aktion (sonst laufen sie ewig zusammen)
- Ein Netzaufruf pro Zug für alle Units
- Training: Target-Netz, Double DQN, Huber-Loss
- Ein Modell pro Pfad (`get_model(path)`): bleibt über alle Spiele, beide Spieler teilen es
- `save()` speichert nur Netz + `steps` + `games` (kein Optimizer, kein Buffer)
- Checkpoints: nach 1.000 / 2.500 / 5.000 / 10.000 Spielen (`CHECKPOINTS`) zusätzlich Kopie in `checkpoints/agent_<spiele>.pth` neben der `.pth`
- Schnittstelle zu `agent.py`:
  - `choose(x, pos, alive, mask, sap_ok, sap_range, explore)` → `(N, 3)` mit `[typ, dx, dy]` pro Unit
  - `train_step(x, pos, alive, actions, reward, next_x, next_pos, next_alive, next_mask, next_sap_ok, sap_range, done)`
    - `x`, `next_x`: `(maps, globals)` vor / nach dem Zug
    - `pos`, `next_pos`: `(N, 2)` · `alive`, `next_alive`: `(N,)` · `actions`: `(N, 3)` · `reward`: `(N,)`
    - `mask`, `next_mask`: `(N, A)` erlaubte Aktionen (`next_mask` für argmax im Target)
    - Trainiert wird jede Unit mit `alive[i]`: Q an `pos[i]`, Target an `next_pos[i]`
    - Unit tot (`not next_alive[i]`) oder `done` → Target nur `reward[i]` (kein Q_target)
    - `next_alive[i]` = dieselbe Unit lebt noch. Stirbt eine Unit und spawnt im selben Zug eine neue mit gleicher ID, ist es False (`World.my_respawned`)
- Target: `r + γ · (1 − done) · Q_target(s′, argmax erlaubter a′)`
- ε abklingen lassen (z. B. 1.0 → 0.05), nicht fest
- Replay-Buffer platzsparend speichern (`float16` / `int8`)