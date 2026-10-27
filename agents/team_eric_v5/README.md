
# V4

Ergänzungen seit V3:
- Replay Buffer (ihr vorschlag)
- Symmetrische Speicherung der Karte

V3:
- Units gehen immer nach unten
- Modell hat nicht mehr gelernt (Q-Wert im späten Training explodiert ab Runde ca. 700)
- Action Mask hat > 90% eingegriffen
- Safety Rules hatten keine Anwendung gefunden

- **Match-Win-Reward zu hoch**: Der Sieg-Bonus überlagert alle anderen Signale.
  - Gegen Default-DQN: Agent gewinnt sowieso, Exploration lohnt sich nicht, also lernt er nichts mehr. 
  - Gegen Kaggle_92: Agent gewinnt fast nie, also kommt kaum ein brauchbares Signal an.
- **Bleibt in der eigenen Hälfte**: Units erkunden die gegnerische Seite kaum. 
- **Units stacken**: Mehrere Units stehen auf demselben Feld.

**Offene Punkte**
- **SAP fehlt**: Die Aktion ist noch nicht implementiert.
- **Nur Spieler 0**: Training läuft bisher nur aus einer Perspektive. Gespiegeltes Training kommt später.

**Ideen / Nächste Schritte**
- **Gegner rotieren**: Regelmäßig wechseln (Kaggle, Default, …).
- **Negative Rewards einführen**: z. B. für Stacking oder Passivität.
- **Langzeit-Training**: Einmal deutlich länger laufen lassen.

# V5

**Ergänzungen seit V4**:
1. Nutze eigene VM (HTW VM kann abgeschaltet werden)
2. **Gegner Rotation** -> Ziel ist besseres diveres Training
3. **Negativer Reward - Stacked** -> Ziel ist Gegner Stacken nicht mehr
4. **Reward: positive_energy_tile**
6. **toward_point_tile auch zu neuen Relics** -> Ziel: Units laufen zum zweiten Relic, sobald es bekannt ist
   - Problem: Ziel waren nur *sichere* Punkt-Felder. Um ein neues Relic ist noch kein Feld sicher -> kein Signal, dorthin zu gehen
   - Jetzt: Ziel = nächstes *freies* Feld, das sicher Punkte bringt **oder** noch ungeprüft im 5x5 um ein Relic liegt (`world.point_tile_candidates()`)
   - Frei = keine andere eigene Unit steht drauf -> sind die Felder an Relic 1 besetzt, ziehen übrige Units zu Relic 2
   - Netz und Features unverändert (gleiche Kanäle), `agent.pth` passt weiter
   - Grenze: bleibt ein Feld an Relic 1 dauerhaft unklar, bleibt es Ziel -> Units können dort hängen. Dann Gewicht von `toward_point_tile` erhöhen (0.1 -> 0.2)

**Einfluss (Erst nach Trainings und Evalphase ausfüllen)**
1. xxx
2. Durch. Punkte pro Match verbessert
3. Units Stacken weniger
4. 
5. 
6. Units erkunden andere Seite

# Nächste Woche

- Impl. Sap
- Training lange durchziehen
- Impl. Mirror Player



**Was erwarten sie bei der Battle Session?**

**Für Sie kaum etwas neues, eher verständnis von Studenten gewährleisten?**

**Eher ein Gesamt-überblick was ich so gemacht habe und was passiert ist, sodass die anderen es verstehen, also tendenziell eher niedrigeres Niveau?**

- Agenten müssen bis mittwoch übernächste woche da sein bis zur battel session (ordner mit Pfad! hochladen)
- ca. 10min. präsentation 
- Wenig zeit für zwischenfragen

vorstufen nicht so im detail (nur was hochstein nicht gesehen hat) - primär nur der den spielt

man soll vertehen wie es funktioniert (was heisst den jetzt replay buffer)
erkenntnisprozess (wenn cool und nötig - ) -> Qwert explosion, nur noch wins, ... | aber kein muss - was hat mich zu fpögender entscheidung geführt - 

Es soll absolut nachvollziehbar sein wie es funktioniert 

Bei Fragen einfach Dr. Hochstein anhauen!!




