wechsel auf CNN Modell - Um Muster in Bildern zu erkennen - Was geht da miteinher?
    vorher: 14 Features pro Unit
    jetzt: ca 15x24x24 Zahlen (Ein einziger Netz aufruf anstatt 16 einzelne Aufrufe)

Ergänzungen:
- Replay Buffer (ihr vorschlag)
- Symmetrische Speicherung der Karte

Letzte Probleme:
- Units gehen immer nach unten
- Modell hat nicht mehr gelernt (Q-Wert im späten Training explodiert ab Runde ca. 700)
- Action Mask hat > 90% eingegriffen
- Safety Rules hatten keine Anwndung gefunden

Reply: Nr 7 nehmen

Aktuelle Probleme:
- Match_win Reward ist zu hoch -> Default DQN = Hört auf zu lernen = Epxplore lohnt nicht mehr, weil ich sowieso immer gewinne; Kaggle_92 = Gewinnt quasi nie
- Bleibt fast immer nur bei der eigenen Hälfte
- Units stacken
- SAP nicht impl.
- Training immer als Spieler 0 (Spiegel soll später kommen)
- Regelmäßiger switch gegen verschiedene Gegner? (Mal Kaggle, mal Default, mal ...) -> Jedes mal Replay Buffer leeren?
negative rewards einbauen
- Mal richtig lange trainieren lassen
- VM - SSH? Torch?