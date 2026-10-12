# LuxAI – Experiment-Logbuch

## Stichwörter in Review verwenden:

**Policy**
- Meine Spielstrategie: „In Situation X mache ich Zug Y.“ 
- Pac-Man: „Geist links → ich laufe nach rechts.“

**Value Function**
- Wie gut ist meine aktuelle Situation? 
- Das ist der erwartete Gesamtgewinn bis Spielende ab diesem Zustand, nicht nur der Gewinn der nächsten Runde. 
- Pac-Man: „Viele Punkte in der Nähe, kein Geist → guter Zustand.“

**Action Value** (Q-Function)
- Wie gut ist dieser bestimmte Zug in meiner aktuellen Situation? 
- Gemeint ist der erwartete Gesamtgewinn, wenn ich jetzt Zug a mache und danach meiner Strategie folge. 
- Pac-Man: „Nach oben gehen = +50, nach links = −500 (Geist).“ 
- Hinweis: Die Q-Function ist nicht selbst das Lernen, sondern das, was gelernt wird. Das Lernen passiert durch das ständige Aktualisieren der Q-Werte.

**Reward Function**
- Das sofortige Feedback des Spiels nach jedem Zug, also Punkte oder Strafe. 
- Pac-Man: Punkt gefressen = +10, Geist gefressen = +200, gestorben = −500. 
- Merke: Reward ist kurzfristig (jetzt). Value und Q sind langfristig (bis Spielende).

**Action Space**
- Alle Züge, die es im Spiel überhaupt gibt. 
- Pac-Man: {oben, unten, links, rechts}. 
- Hinweis: Die Regeln bestimmen, welche davon gerade erlaubt sind. Eine Wand blockiert zum Beispiel „links“. Das nennt man „legale Aktionen“ oder Action Masking.

**State Space vs. Observation Space**
- State: Die komplette Wahrheit über das Spiel, also alles, was es gibt. 
- Observation: Das, was der Agent davon sieht und für seine Entscheidung nutzt. 
- Bei Schach oder Pac-Man sind beide gleich, weil alles sichtbar ist. Bei Poker unterscheiden sie sich: Der State enthält alle Karten, die Observation nur die eigenen und die offenen.

**Zusammenhang in einem Satz:**

Der Agent sieht eine **Observation**, wählt per **Policy** eine Aktion aus dem **Action Space**, bekommt dafür einen **Reward** und verbessert damit seine Schätzungen **(Q / Value)**, die wiederum die Policy verbessern.


## Logging

| Datum | Hypothese | Änderung | Winrate | Commit |
|-------|-----------|----------|---------|--------|
| 09.26.10 | ? | ? | ? | ? |

---

## WinRate Ausgangslage

Agent spielt 20 Spiele gegen Kaggle_92

## W1 - 09.26.10

**Input (Features)**
- ...
- ...

**Output (Actions)**
- ...
- ...

**Policy**
- ...
- ...

**Training**
- ...
- ...

**Änderungen zu letzter Woche**
- ...
- ...

**Winrate Vergleich**: XX zu XX (09.10.26)

**Beobachtungen / Schwächen**
- ...
- ...

---

...
