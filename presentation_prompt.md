# Rolle
Du bist ein erfahrener ML-Engineer und Technical Writer. Du erklärst komplexe
Reinforcement-Learning-Systeme so, dass man sie in wenigen Minuten versteht.

# Aufgabe
Analysiere den LuxAI-Agenten im Ordner
/Users/ericg/git/aki_luxai/02_LuxAI/agents/team_eric_v4
und erstelle EINE einzelne, scrollbare HTML-Seite. Sie hat zwei Teile:
1. Die Architektur des Agenten, einfach erklärt nach dem KISS-Prinzip.
2. Die Entwicklung des Agenten, belegt mit den Daten aus der Logging-CSV.

# Zielgruppe
Ein Dozent. Er kennt Machine Learning, aber nicht diesen Code.
Er will schnell verstehen:
- Welche Daten gehen wo rein?
- Was passiert damit?
- Was kommt raus?
- Wie hat sich der Agent über die Spiele entwickelt?

# Sprache
- Deutsch.
- Fachbegriffe (z. B. Q-Wert, Epsilon, Replay Buffer, Action Mask, Loss) dürfen
  bleiben. Erkläre jeden beim ersten Auftreten in einem kurzen Halbsatz oder als
  Tooltip.
- Kurze Sätze, kein Marketing-Ton.

# Vorgehen
1. Lies zuerst den Code und finde heraus, wie der Agent wirklich aufgebaut ist:
    - Einstiegspunkt
    - Feature-Erzeugung
    - Modell bzw. Netz
    - Action-Masking
    - Aktionswahl
    - Reward-Berechnung
    - Training-Loop
    - Laden und Speichern der .pth-Datei
    - Logging
      Nimm keinen Algorithmus an. Stelle ihn anhand des Codes fest.
      Die CSV enthält q, epsilon und lernschritte. Prüfe also, ob es z. B. DQN ist.
2. Finde die Logging-CSV im Ordner. Finde im Code heraus, wo und wie jede Spalte
   geschrieben wird, damit du ihre Bedeutung korrekt erklärst.
3. Erst danach baust du die Seite.

Regel: Alles auf der Seite muss aus Code oder CSV belegbar sein. Wenn du etwas
nicht eindeutig findest, schreibe „nicht eindeutig im Code“, statt zu raten.

# Teil 1: Architektur

## Gesamtbild (ganz oben)
- Ein Satz: Was ist dieser Agent?
- Ein Boxen-und-Pfeile-Diagramm des gesamten Datenflusses:
  Observation → Features → Modell → Action-Masking → Aktion → Umgebung
  → Reward → Training-Loop → zurück ins Modell
- Gibt es im Code weitere relevante Bausteine (z. B. Replay Buffer,
  Target-Netz, Heuristik-Fallback), ergänze sie im Diagramm.
- Beschrifte die Pfeile mit der Art der Daten, die fließen
  (z. B. „Feature-Tensor“, „5 Q-Werte“, „Maske“, „Reward-Skalar“).

## Ein Abschnitt pro Baustein, immer gleich aufgebaut
Oberste Ebene, sichtbar ohne Klick, insgesamt 2–3 Sätze:
- REIN: Welche Daten kommen an?
- PASSIERT: Was macht der Baustein damit?
- RAUS: Was gibt er weiter, und an wen?

Dazu ein kleines Mini-Diagramm (rein → Box → raus).

Ausklappbar (<details>) für alle, die tiefer wollen:
- Tensor-Shapes
- vollständige Feature-Liste
- Layer und Größen
- Reward-Komponenten mit Gewichten
- Masken-Regeln
- relevante Hyperparameter (z. B. Epsilon, Lernrate, Batch-Größe)
- Pfad zu Code-Ausschnitten nur, wenn er beim Verständnis hilft

## Mindestens diese Bausteine (soweit im Code vorhanden)
1. Observation: Was liefert die LuxAI-Umgebung pro Schritt?
2. Features: Wie wird daraus der Modell-Input?
3. Modell: Architektur, Input und Output.
4. Action-Masking: Regeln no_energy, blocked, stacked. Was bedeuten sie?
5. Aktionen: stay, up, right, down, left. Wie wird gewählt
   (z. B. Epsilon-greedy)?
6. Reward: Komponenten on_point_tile, team_gain, near_relic, explore und wie sie
   zum Gesamt-Reward werden.
7. Training-Loop: Wann wird gelernt? Was ist ein Lernschritt? Wann wird die .pth
   geladen oder gespeichert?

# Teil 2: Entwicklung (Benchmarks aus der CSV)

## CSV-Format
Eine Zeile pro Spiel. Spalten:
spiel, siege, niederlagen, punkte, relics, punktefelder, maske_quote,
regel_no_energy, regel_blocked, regel_stacked,
aktion_stay, aktion_up, aktion_right, aktion_down, aktion_left,
reward, reward_on_point_tile, reward_team_gain, reward_near_relic, reward_explore,
loss, q, lernschritte, epsilon, pth_geladen

- Überspringe unvollständige oder kaputte Zeilen (z. B. eine abgeschnittene
  letzte Zeile).
- Nenne die Zahl der übersprungenen Zeilen in einer kleinen Fußnote.
- Bette die bereinigten Daten direkt in die HTML ein, damit die Seite offline
  funktioniert.

## Oben: Kennzahlen-Kacheln
- Anzahl Spiele
- Gesamt-Siegquote
- Siegquote der letzten 10 % der Spiele im Vergleich zu den ersten 10 %
- Ø Punkte
- Ø Reward
- Gesamtzahl der Lernschritte

## Danach: Diagramme
Immer mit der Spielnummer auf der x-Achse. Rohwerte zeigst du blass, darüber einen
gleitenden Durchschnitt (Fenster sinnvoll zur Datenmenge wählen und angeben).
1. Siegquote pro Spiel: siege / (siege + niederlagen)
2. Punkte, Relics, Punktefelder
3. Reward gesamt und seine vier Komponenten (gestapelt oder als kleine
   Einzeldiagramme)
4. Loss und Q-Wert
5. Aktionsverteilung über die Zeit: stay, up, right, down, left, gestapelt
   auf 100 %
6. Action-Masking: maske_quote und die drei Regel-Anteile
7. Epsilon und Lernschritte

Jedes Diagramm bekommt:
- einen klaren Titel
- Achsenbeschriftungen mit Einheit
- eine Bildunterschrift in einem Satz, die nur beschreibt, WAS gezeigt wird

KEINE Interpretation. Also keine Sätze wie „weil Reward X eingeführt wurde …“.

# Design
- Eine einzige, selbständige HTML-Datei: CSS und JS inline.
- Keine externen Abhängigkeiten. Diagramme als inline SVG oder Canvas mit
  Vanilla JS.
- Ruhig und klar: viel Weißraum, eine Akzentfarbe, gut lesbare Schrift.
- Eine einheitliche Farbe pro Baustein, die in Diagrammen und Abschnitten
  wiederkehrt.
- Oben eine kleine Navigation (Sprungmarken) zu „Gesamtbild“, den Bausteinen und
  „Entwicklung“.
- Funktioniert auf Laptop und Beamer. Kein horizontales Scrollen.

# Nicht auf die Seite
- Interne Pfade, Benutzernamen, Rechnernamen
- Auskommentierter oder toter Code
- Unfertige Experimente
- Alles, was nichts mit der Funktion des Agenten zu tun hat

# Dateien
- Ändere keine bestehende Datei.
- Lege nur die neue HTML-Seite an: agent_overview.html im selben Ordner.

# Qualitätscheck vor der Abgabe
1. Rechne jede Kennzahl-Kachel einmal unabhängig aus der CSV nach.
2. Prüfe, dass jeder Name im Diagramm (Features, Layer, Reward-Komponenten,
   Masken-Regeln) so im Code existiert.
3. Öffne die Seite im Browser. Prüfe:
    - Alle Diagramme rendern.
    - <details> klappen auf.
    - Kein horizontales Scrollen.
    - Keine Konsolenfehler.
4. Lies die oberste Ebene jedes Bausteins noch einmal. Ist sie wirklich in
   2–3 Sätzen „rein / passiert / raus“? Wenn nicht, kürzen.

# Abgabe
Die fertige HTML-Datei. Dazu höchstens 3 Sätze:
- Was die Seite enthält
- Welche Punkte im Code nicht eindeutig waren