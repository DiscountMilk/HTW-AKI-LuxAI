# Ideen

- Map muss nur bis zur hälfte erkundet werden, dann hat man die ganze Map
- Logging im NICHT-Training (Logging ohne Epsilon-Greedy)
- Lösung finden: Units laufen nur nach unten 

# Daten

- Welche Daten bekomme ich aus OBS und wie? - Diesen Daten aktualisieren sich jedes runde? 
- Welche Daten bekomme ich aus env_cfg und wie? - Muss das jedes mal neu init werden?

## Reward Design

?

# Safety Rules
_Strategische Pflichtentscheidungen (nur für Fälle, in denen du dir sicher bist, dass die Regel die beste Entscheidung trifft und das Netz dort nichts lernen muss.)_


# Action Mask
_Verringert die möglichen Aktionen, die das Netz wählen kann (für Regeln, die nur Unsinn verbieten.)_



# Features

...

# Architecture

`agent.py`: Wie läuft ein Schritt ab? Runner-Schnittstelle, nur Aufrufe
`agent_world.py`: Was weiß ich? obs -> Wissen, plus Spielkonstanten
`agent_rules.py`: Was MUSS / DARF ich? safety_rules + action_mask, zwei Listen
`agent_brain.py`: Was lerne ich? Features + Netz + Training
`training_log`: Logger

# Logging

- Welche Regeln werden wie oft verwendet: SafetyRules-Regeln
- Welche Regeln werden wie oft verwendet: ActionMask-Regeln
- Welche Reward Regeln, bringt wie viel?

`ERGEBNIS`
- `Matches 1:2`: Gewonnene zu verlorenen Matches in diesem Spiel.
- `Punkte 172`: Relic-Punkte deines Teams, summiert über alle 3 Matches.
- `Relics gefunden 2`: Wie viele Relic Nodes deine Einheiten in diesem Spiel gesehen haben.
- `Punktefelder erkannt 4`: Felder, die World am Spielende sicher als Punktefeld kennt.

`SAFETY RULES`
- `Regeln 0% | Netz 100%`: Wer hat entschieden
- ... regeln?

`ACTION MASK`
- `no_energy_to_move 0.44`: Pro Netz-Entscheidung verbietet diese Regel im Schnitt 0.44 der 4 Bewegungen. Hat eine Einheit keine Energie, verbietet die Regel alle 4. 0.44 bedeutet also, dass in etwa 11 % der Entscheidungen die Energie fehlte (0.44 / 4).
- `blocked 1.18: Im Schnitt sind 1.18 von 4 Bewegungen durch Kartenrand oder Asteroid blockiert. Verbieten beide Regeln dieselbe Bewegung, zählt sie bei beiden.

`BRAIN`
- `Loss 0.056`: Durchschnittlicher Lernfehler aller Lernschritte in diesem Spiel. Gemessen wird, wie weit die Vorhersage Q(s, a) vom Ziel r + γ · max Q(s2) entfernt ist. Bei DQN muss der Loss nicht sinken, er darf nur nicht dauerhaft wachsen.
- `Q 3.43`: Durchschnittlicher Q-Wert der gewählten Aktionen beim Lernen, also die erwartete Summe aller künftigen Rewards, wobei spätere schwächer zählen. Er steigt, während das Netz lernt, dass sich Punktefelder lohnen. Ist er stabil, liegt er etwa bei Reward / (1 − γ).
- `Epsilon 0.1`: 10 % der Netz-Entscheidungen sind zufällig, damit das Netz Neues ausprobiert.
- `Aktionen ...`: Wie oft das Netz in diesem Spiel welche Bewegung gewählt hat. Ein hoher stay-Anteil ist normal, weil Stehenbleiben auf einem Punktefeld Punkte bringt.
- `Ø10: ...`: Schnitt der letzten 10 Spiele, in Klammern die Änderung.

`REWARD`
- 
