# Daten

Welche Daten bekomme ich aus OBS und wie? - Diesen Daten aktualisieren sich jedes runde?

Welche Daten bekomme ich aus env_cfg und wie? - Muss das jedes mal neu init werden?

# Regel

kann OBS und env_cfg Daten direkt nutzen

# RL

Muss OBS und env_cfg Daten in Bildartigen Tenso umwandeln (feature engineering)

## Reward Design

?

# Meine Strategy

- `Safety_rules`: Strategische Pflichtentscheidungen (nur für Fälle, in denen du dir sicher bist, dass die Regel die beste Entscheidung trifft und das Netz dort nichts lernen muss.)
- `action_mask`: Verringert die möglichen Aktionen, die das Netz wählen kann (für Regeln, die nur Unsinn verbieten.)

# Architecture

`agent.py`: Wie läuft ein Schritt ab? Runner-Schnittstelle, nur Aufrufe
`world.py`: Was weiß ich? obs -> Wissen, plus Spielkonstanten
`rules.py`: Was MUSS / DARF ich? safety_rules + action_mask, zwei Listen
`brain.py`: Was lerne ich? Features + Netz + Training



