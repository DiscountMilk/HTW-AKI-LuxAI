# Ideen
_Was könnte verändert werden zur nächsten Review Session_

- Map muss nur bis zur hälfte erkundet werden, dann hat man die ganze Map
- Logging im NICHT-Training (Logging ohne Epsilon-Greedy)
- Lösung finden: Units laufen nur nach unten
- UNITS data bleibt bestehen mit Attribut "ALIVE=TRUE" -> mehr reward, wenn ALIVE bleibt 
- Model entscheidet bisher pro Unit - Wie wäre es, wenn das Modell immer alle Informationen ds bretts hat und immer eine Entscheidung für alle trifft und nicht pro Unit

_Hier nichts ändern, mache ich selbst!!_


# Umgesetzt

_Hier nichts ändern, mache ich selbst_

_Hier kommt: Was wurde verändert seit dem letzten Review + Was hat es gebracht_

# Daten
_Welche Daten entstehen **wo** und werden **wie** in `world.py` verarbeitet_

- **`env_cfg`** kommt einmal pro Spiel (`set_env_cfg`). Feste Spielregeln dieses Spiels → in `World.__init__` gespeichert, ändern sich nicht mehr.
- **`obs`** kommt jeden Schritt (`act`, `update`). Nur das Sichtfeld, Unsichtbares ist `-1` → `World.update()` arbeitet es ein.
- **`World`** = Gedächtnis: merkt sich, was einmal gesehen wurde, spiegelt es auf die andere Kartenhälfte und berechnet, was `obs` nicht direkt liefert.
- Feste Werte, die für jedes Spiel gelten (Kartengröße, Max-Energie, Aktionen …), stehen als Konstanten oben in `world.py`.

| Quelle | Rohdaten | In `World` als | Verarbeitung |
|---|---|---|---|
| `env_cfg` | `unit_move_cost`, `unit_sap_cost`, `unit_sap_range`, `unit_sensor_range`, `match_count_per_episode` | gleichnamige Attribute | 1× pro Spiel übernommen |
| Agent | `player` | `team_id`, `opp_id` | 0 / 1, Index in alle obs-Arrays |
| `obs` | `steps`, `match_steps` | `steps`, `match_steps` | direkt |
| `obs` | `team_points` | `team_points` | eigenes Team, aktuelles Match |
| `obs` | `team_wins` | `team_wins`, `opp_wins` | beide Teams |
| `obs` | `units_mask`, `units.position`, `units.energy` | `units` (Liste `Unit(id, pos, energy)`) | nur Zeilen mit Maske = True |
| `obs` | `sensor_mask`, `map_features.tile_type`, `map_features.energy` | `tile_type`, `tile_energy` | nur sichtbare Felder überschreiben **+ Spiegelfeld** |
| `obs` | `relic_nodes`, `relic_nodes_mask` | `relics` (Set) | nie entfernt **+ Spiegel-Relic** |
| **berechnet** | `team_points` jetzt − vorher | `point_gain` | 0 bei `match_steps == 0` (neues Match) |
| **berechnet** | Summe aller `point_gain` | `total_points` | Punkte über alle Matches |
| **berechnet** | Schritt der letzten Sichtung | `last_seen` | wie alt ist das Wissen über ein Feld |
| **berechnet** | `point_gain` + Units nahe Relic | `point_tile` (1 / 0 / −1) | Punktefelder erschließen **+ Spiegelfeld**; neue Relic setzt „0“ in ihrem 5×5 zurück auf −1 |

Spiegelung: Feld `(x, y)` = Feld `(23 − y, 23 − x)` – gilt für Feldtypen, Energie, Relics und Punktefelder.

**==Frage: gibt es in OBS oder env_cfg Werte, die nicht weiterverwendet werden in world.py?==**

# Action Mask

- Die Action Mask **verbietet Unsinn**, bevor das Netz wählt. Das Netz entscheidet nur unter den erlaubten Aktionen – es muss Unsinn also nicht erst lernen.
- `allowed(unit)` fragt für jede der 5 Aktionen alle Regeln. Verbietet **irgendeine** Regel die Aktion, ist sie verboten. Ergebnis: `bool[5]`, Index = STAY, UP, RIGHT, DOWN, LEFT.
- Technisch: verbotene Aktionen bekommen beim Wählen Q = −∞. **Die Reihenfolge der Regeln ist egal** – es werden immer alle geprüft. Ist alles verboten, bleibt STAY erlaubt.
- Neue Regel: Methode `(unit, action) -> True = verboten` schreiben und in `self.rules` eintragen.

| Regel | Verbietet | Warum |
|---|---|---|
| `no_energy` | jede Bewegung, wenn Energie < `unit_move_cost` | Zug würde scheitern |
| `blocked` | Bewegung aus der Karte oder in einen (zuletzt gesehenen) Asteroiden | Energie weg, keine Bewegung |
| `stacked` | STAY, wenn mehrere eigene Units auf dem Feld stehen – außer für die kleinste ID | Feld zählt nur 1×, Stapel = leichtes Sap-Ziel |

# Features

- Features = **was das Netz sieht**: World + Unit → Vektor `s_t` mit Zahlen etwa zwischen −1 und 1.
- Jedes Feature ist eine Methode `(unit, w) -> Liste von Zahlen` in `FeatureEngineering`.
- **Hinzufügen:** Methode schreiben → mit Anzahl Zahlen in `PARTS` eintragen → alte `agent.pth` umbenennen.
- **Entfernen:** Zeile aus `PARTS` löschen (Methode darf bleiben) → alte `agent.pth` umbenennen.
- `SIZE` (= Eingang des Netzes) rechnet sich selbst aus `PARTS`. Reihenfolge in `PARTS` = Reihenfolge im Vektor – nach dem Training nicht mehr umsortieren.

| Feature | Zahlen | Aus `World` | Bedeutung |
|---|---|---|---|
| `energy` | 1 | `unit.energy` / 400 | eigene Energie |
| `match_progress` | 1 | `match_steps` / 100 | früh erkunden, spät punkten |
| `tile_energy` | 1 | `tile_energy[x, y]` / 20 | Energie des eigenen Felds |
| `to_relic` | 2 | `relics` | Richtung (dx, dy) zur nächsten Relic |
| `to_point_tile` | 2 | `point_tile == 1` | Richtung zum nächsten Punktefeld |
| `on_point_tile` | 1 | `point_tile[x, y]` | steht auf Punktefeld? |
| `near_relic` | 1 | `near_relic()` | im 5×5 um eine Relic? |
| `to_unexplored` | 2 | `relics` + `point_tile == -1` | Richtung zum nächsten ungetesteten Relic-Feld |
| `stacked` | 1 | `units` | andere eigene Units auf meinem Feld (0 … 1) |
| **Summe** | **12** | | = `FeatureEngineering.SIZE` |

Richtungen sind durch die Kartenbreite geteilt; kein Ziel bekannt → (0, 0).

## Reward Design

- Reward = **Rückmeldung nach jedem Schritt**, wie gut die Aktion einer Unit war. Das Netz lernt Aktionen, die langfristig viel Reward bringen (Q-Wert).
- Reward = **Summe aus gewichteten Termen**. Jeder Term ist eine Methode `(unit, w) -> Zahl`, das Gewicht steht in `self.terms`.
- **Reihenfolge egal** – es wird nur addiert.
- Neuer Term: Methode schreiben → mit Gewicht in `self.terms` eintragen. `agent.pth` darf bleiben (Netzgröße ändert sich nicht).
- Ein Reward braucht **kein eigenes Feature**. Das Netz muss aber *sehen können*, wovon der Reward abhängt – sonst kann es nicht lernen, wie es ihn bekommt.

| Term | Gewicht | Wert | Soll bewirken | Feature, mit dem das Netz es sieht |
|---|---|---|---|---|
| `on_point_tile` | 2.0 | 1 auf Punktefeld, geteilt durch Units am Feld | auf Punktefeldern stehen, nicht stapeln | `on_point_tile`, `to_point_tile`, `stacked` |
| `team_gain` | 0.03 | Punktezuwachs des Teams | auch noch unbekannte Punktefelder nutzen | keins direkt (`near_relic`, `to_unexplored` helfen) |
| `near_relic` | 0.05 | 1 im 5×5 um eine Relic | zu Relics gehen | `near_relic`, `to_relic` |
| `explore` | 0.1 | 0 … 1: viel Karte im Sichtfeld, wenig eigene Units darin | von Wand und Kameraden weg, mehr erkunden | `stacked` (nur teilweise) – **Lücke:** kein Feature für Wand-/Mittenabstand |
| `positive_energy_tiles` | – | noch leer (`pass`) | – | nicht in `self.terms`, daher inaktiv |

# Network Architecture

```
s_t (12 Features) → Linear 12→128 → ReLU → Linear 128→128 → ReLU → Linear 128→5 → Q(s_t, a) für STAY, UP, RIGHT, DOWN, LEFT
```

- **DQN:** Das Netz schätzt für jede Aktion den Q-Wert = erwartete Summe künftiger Rewards. Gewählt wird die erlaubte Aktion mit dem höchsten Q-Wert (ε-greedy).
- **Ein Netz für alle Units:** jede Unit entscheidet einzeln, alle teilen dieselben Gewichte.
- **Lernziel (TD):** `Q(s, a)` soll `r + γ · max Q_target(s′)` werden.

| Parameter | Wert | Was er tut |
|---|---|---|
| `learning_rate` | 1e-3 | Schrittgröße beim Anpassen der Gewichte (Adam) |
| `gamma` (γ) | 0.95 | wie stark künftige Rewards zählen; 0.95 ≈ 20 Schritte Horizont |
| `epsilon` (ε) | 0.1 | 10 % zufällige Aktionen im Training (Exploration), 0 in der Evaluation |
| Replay Buffer | 100 000 | merkt sich Erfahrungen `(s, a, r, s′)`; älteste fliegt raus |
| `batch_size` | 64 | so viele zufällige Erfahrungen pro Lernschritt – zufällig gemischt, damit das Netz nicht nur die letzten Schritte lernt |
| `target_sync` | 500 | Target-Netz = langsame Kopie des Netzes, alle 500 Lernschritte nachgezogen – sonst jagt das Netz sein eigenes, wanderndes Ziel |
| Loss | Huber (`smooth_l1`) | Lernfehler, robuster gegen Ausreißer als MSE |
| `pending` | – | merkt pro Unit `(s, a)` bis zum nächsten Schritt, dann wird daraus mit Reward und `s′` eine Erfahrung |
