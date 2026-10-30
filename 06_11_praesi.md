# Kontext
Ich (Eric) habe für einen Kurs einen Agenten für LuxAI (Season 3) mit Reinforcement
Learning gebaut. Ich halte dazu eine 10-minütige Abschlusspräsentation vor
Mitstudierenden (der Dozent ist auch dabei). Du erstellst diese Präsentation
direkt aus meinem Code und meinen Logs.

Agenten-Ordner: /Users/ericg/git/aki_luxai/02_LuxAI/agents/team_eric_v5

Die Präsentation ist eine kompakte Version der offiziellen Kursanforderungen.
Diese Punkte müssen alle vorkommen, aber knapp:
- Algorithmus als Pseudocode
- Architektur des neuronalen Netzes
- RL-Algorithmus
- Trainingskonfiguration
- Feature Engineering
- Trainingsablauf (Anzahl Spiele, Gegner)
- Ausgewählte Spiele
- Beobachtungen, Hypothesen, Erkenntnisse
- Fragen
- Next Steps

# Rolle
Du bist ein erfahrener ML-Engineer und Technical Writer. Du erklärst
RL-Systeme so einfach wie möglich, aber fachlich korrekt (KISS).

# Ziel
Die Mitstudierenden sollen nach 10 Minuten verstehen, wie der Agent funktioniert,
und die wichtigsten Ideen selbst nachbauen können:
- Welche Daten gehen wo rein?
- Was passiert damit?
- Was kommt raus?
- Wie hat sich der Agent über die Trainingsspiele entwickelt?
- Was haben wir daraus gelernt?

# Vorgehen (in dieser Reihenfolge)

## Schritt 1: Code verstehen (noch nichts bauen)
Lies den Code vollständig. Stelle fest, wie der Agent wirklich aufgebaut ist.
Nimm keinen Algorithmus an, sondern belege ihn mit dem Code.
Die CSV enthält Spalten wie q, epsilon und lernschritte. Prüfe also, ob es
z. B. DQN, Double DQN oder eine andere Variante ist und woran man das erkennt.

Kläre für jeden Baustein: Datei/Funktion, Input (mit Shape), Verarbeitung, Output.
- Einstiegspunkt (welche Funktion ruft das Spiel pro Zug auf?)
- Zustands-/Kartenspeicher (Map-Memory)
- Feature-Erzeugung (welche Kanäle/Features, welche Shape?)
- Modell (Layer, Größen, Output)
- Aktionswahl (Epsilon-Greedy? Action Mask? welche Regeln?)
- Reward-Berechnung (jede Komponente mit Gewicht)
- Training-Loop (Replay Buffer, Batchgröße, Target-Netz, Loss, Optimizer,
  Lernrate, Gamma, Update-Frequenz)
- Gegner-Auswahl im Training
- Logging

## Schritt 2: Daten verstehen
- Finde die Logging-CSV(s). Gibt es mehrere, nimm die zum aktuellen Agenten
  gehörende und nenne in der Abgabe, welche du genommen hast.
- Finde für jede Spalte die Code-Stelle, an der sie geschrieben wird. Erkläre
  die Bedeutung erst danach (z. B. ist q ein Mittelwert, ein Maximum, pro Zug
  oder pro Spiel?).
- Ermittle aus der CSV: Anzahl Spiele, Lernschritte, Gegner pro Phase,
  Ergebnisse gegen jeden Gegner.
- Suche nach Replay-Dateien (z. B. JSON/HTML-Replays). Gibt es keine, wähle
  die Beispiele stattdessen aus auffälligen CSV-Zeilen und sag das offen.

## Schritt 3: Präsentation bauen
Erst wenn Schritt 1 und 2 abgeschlossen sind.

# Aufbau (12–14 Hauptslides, ~45 s pro Slide, plus Backup)

1. **Titel**: Ein Satz: Was ist dieser Agent?
2. **Gesamtbild**: Boxen-und-Pfeile-Diagramm vom Spielzustand bis zur Aktion,
   inklusive Trainingsschleife. Pfeile mit dem beschriften, was fließt
   (z. B. „Feature-Tensor 12×24×24"). Dieses Diagramm erscheint auf jeder
   Baustein-Slide klein mit hervorgehobenem Baustein („Sie sind hier").
3. **Algorithmus als Pseudocode**: Der komplette Ablauf eines Zuges und eines
   Trainingsschritts in max. 15 Zeilen. Jede Zeile entspricht einem Baustein
   und trägt dessen Farbe.
4. **Feature Engineering**: welche Kanäle/Werte, als Gitter oder Liste
   visualisiert, jeweils mit „warum".
5. **Map-Memory und Mirror-Logik**:
   - Wie die Einheiten ein gemeinsames Kartenbild aufbauen, wie Spiegelung
     genutzt wird und wann Tiles „vergessen" werden (genaue Logik im Code
     prüfen, z. B. nach wie vielen Zügen).
   - Wie der Agent dafür sorgt, dass es egal ist, ob er player_0 oder
     player_1 ist. Vorher/Nachher-Grafik.
6. **Netzarchitektur (CNN)**: Layer-Folge mit Shapes als Diagramm.
7. **RL-Algorithmus und Aktionswahl**: Welcher Algorithmus (mit Beleg),
   Epsilon-Greedy, Action Mask mit ihren Regeln.
8. **Rewards**: jede Komponente mit Gewicht und wofür sie belohnt/bestraft.
9. **Training (1–2 Slides, wichtigster Teil)**: Ich verstehe es selbst nicht
   ganz. Erkläre es als Kreislauf und verfolge EIN konkretes Beispiel durch
   alle Stationen:
   Zug spielen → Erfahrung (s, a, r, s') speichern → Batch ziehen →
   Ziel-Q-Wert berechnen → Loss → Gewichte anpassen → Epsilon senken.
   Zeige die Kernformel einmal, mit jeder Variable in Worten erklärt.
10. **Trainingskonfiguration und Gegner**: Kompakte Tabelle mit den
    wichtigsten 6–8 Werten (Spiele gesamt, Lernschritte, Batchgröße, Lernrate,
    Gamma, Epsilon-Verlauf, Buffergröße) und die Gegner-Rotation als
    Zeitleiste: gegen wen wurde wann trainiert und warum.
11. **Entwicklung**: 2–3 Liniendiagramme aus der CSV (z. B. Ergebnis/Winrate,
    Q-Wert, Epsilon über Spiele), geglättet mit gleitendem Mittelwert, dazu
    3–4 Kennzahl-Kacheln. Jeweils ein Satz: Was sieht man?
12. **Ausgewählte Spiele**: 1 positives und 1 negatives Spiel. Jeweils:
    was ist passiert, warum (welcher Baustein erklärt es).
13. **Beobachtungen, Hypothesen, Erkenntnisse**: Drei klar getrennte Spalten.
    Beobachtung = belegt durch Daten. Hypothese = deine Erklärung dafür,
    als Hypothese gekennzeichnet. Erkenntnis = was man beim Nachbauen
    beachten sollte.
14. **Next Steps und Fragen**: 3–4 konkrete nächste Schritte, die sich aus den
    Hypothesen ergeben. Darunter groß „Fragen?".

**Backup (nach der letzten Slide, nicht in der Zeit):** vollständige
Hyperparameter-Tabelle, vollständige Feature-Liste, Reward-Details.
Diese Slides sind als „Backup" markiert und dienen für Rückfragen.

Baustein-Slides folgen immer demselben Muster:
- Oben max. 2–3 Sätze nach dem Schema „rein → passiert → raus".
- Darunter eine Grafik, die zeigt, was passiert und warum.

# Sprache
- Deutsch, kurze Sätze, kein Marketing-Ton.
- Fachbegriffe (Q-Wert, Epsilon, Replay Buffer, Action Mask, Loss, Target-Netz,
  Gamma) bleiben, werden aber beim ersten Auftreten in einem kurzen Halbsatz
  direkt auf der Slide erklärt. Keine Tooltips, die funktionieren am Beamer nicht.
- Keine Zahl, kein Name und keine Behauptung ohne Beleg im Code oder in der CSV.
  Ist etwas unklar, schreib es nicht auf die Slide, sondern nenne es in der Abgabe.
- Hypothesen immer als solche kennzeichnen, nie als Fakt formulieren.

# Design und Technik
- Eine einzige, selbständige HTML-Datei, CSS und JS inline,
  keine externen Abhängigkeiten (auch keine Fonts oder CDNs).
- CSV-Daten, die für Diagramme gebraucht werden, als JS-Array einbetten.
- Diagramme als inline SVG oder Canvas mit Vanilla JS.
- Navigation: Pfeiltasten, Leertaste, Klick-Buttons vor/zurück,
  Slide-Nummer „3 / 14" (Backup-Slides zählen nicht mit), Fortschrittsbalken.
  Slide-Nummer in der URL (#3), damit Neuladen auf derselben Slide bleibt.
- 16:9, skaliert auf Laptop und Beamer, Schrift aus der letzten Reihe lesbar.
  Starker Kontrast, heller Hintergrund.
- Ruhig: viel Weißraum, eine Akzentfarbe, eine gut lesbare Systemschrift.
- Jeder Baustein hat eine feste Farbe, die im Gesamtdiagramm, im Pseudocode,
  auf seiner Slide und in allen Grafiken gleich bleibt.
- Pro Slide höchstens eine Kernaussage.

# Nicht auf die Slides
- Interne Pfade, Benutzernamen, Rechnernamen
- Auskommentierter oder toter Code, unfertige Experimente
- Alles, was nichts mit der Funktion des Agenten zu tun hat

# Dateien
- Ändere keine bestehende Datei.
- Lege nur eine neue Datei an: praesi.html im Agenten-Ordner.

# Qualitätscheck vor der Abgabe
1. Rechne jede Kennzahl unabhängig ein zweites Mal aus der CSV nach
   (z. B. mit einem kurzen Python-Skript, das du danach nicht speicherst).
2. Prüfe, dass jeder Name auf den Slides (Features, Layer, Reward-Komponenten,
   Masken-Regeln, Hyperparameter, Gegner) genau so im Code existiert.
3. Prüfe, dass der Pseudocode zum echten Ablauf im Code passt.
4. Prüfe, dass jeder Pflichtpunkt aus dem Kontext-Abschnitt auf einer
   Hauptslide vorkommt.
5. Öffne die Seite im Browser und prüfe: alle Diagramme rendern, keine
   Konsolenfehler, Navigation funktioniert in beide Richtungen, kein Text
   läuft über den Slide-Rand.
6. Lies den oberen Text jeder Baustein-Slide. Ist er wirklich 2–3 Sätze
   „rein → passiert → raus"? Wenn nicht, kürzen.
7. Zähle die Hauptslides. Passt das in 10 Minuten? Wenn nicht, zusammenfassen
   oder Details ins Backup verschieben.

# Abgabe
Die fertige praesi.html. Dazu höchstens 3 Sätze:
welche Punkte im Code nicht eindeutig waren und welche CSV/Replays du verwendet hast.