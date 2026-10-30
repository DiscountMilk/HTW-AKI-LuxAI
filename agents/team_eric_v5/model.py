"""
model.py = CNN, Aktionswahl, Training, Replay-Buffer, Speichern.
Kennt die World NICHT, bekommt nur Features (x), Positionen und Masken.

──────────────────────────────────────────────────────────────────────
IDEE
    Ein vollständig faltendes Netz: Karte rein, Karte raus.
        Eingang  (C + G, W, H)   Karten-Features + globale Werte als konstante Kanäle
        Ausgang  (A + 1, W, H)   Q-Wert pro Bewegung und Feld + Sap-Karte
    Q-Werte einer Bewegung = Ausgang an der Position der Unit (x, y).
    Sap-Karte (letzter Kanal): Q für "Sap auf DIESES Feld", gelesen am ZIELFELD.
    Eine Unit vergleicht ihre 5 Bewegungen mit den Sap-Werten aller Felder in Reichweite.
    Ein Netzaufruf pro Zug für alle Units.

    Dilatierte Faltungen (Abstand 1, 2, 4, 8, 16, 1): jedes Feld "sieht" die ganze Karte,
    obwohl es nur 6 Schichten sind.

TRAINING (Double DQN)
    Q(s, a)  soll  r + γ · (1 − ende) · Q_target(s′, argmax erlaubter a′ von Q(s′))  werden
    ende = Unit tot / neu gespawnt oder Spiel vorbei.
    Ein Buffer-Eintrag = ein Zug mit allen Units.

BENUTZUNG
    get_model(path)  statt Model(): der Runner baut Agents jedes Spiel neu,
                     das Modell (Netz + Buffer + ε) soll aber bleiben. Selbstspiel teilt ein Modell.
──────────────────────────────────────────────────────────────────────
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn

from agents.team_eric_v5.features import FeatureEngineering

# ---------- Aktionen ----------
N_ACTIONS = 5               # Bewegungen: 0 stay, 1 up, 2 right, 3 down, 4 left
SAP = 5                     # Aktions-Typ Sap = Kanal der Sap-Karte

# ---------- Netz ----------
HIDDEN    = 32              # Kanäle in den versteckten Schichten
DILATIONS = (1, 2, 4, 8, 16, 1)

# ---------- Training ----------
GAMMA         = 0.98        # Blick in die Zukunft, ca. 50 Züge
LEARNING_RATE = 3e-4
BATCH_SIZE    = 32          # Züge pro Lernschritt (je Zug mehrere Units)
BUFFER_SIZE   = 10_000      # Züge im Replay-Buffer (ca. 16 Spiele Selbstspiel)
TRAIN_EVERY   = 4           # nur jeden 4. Zug lernen (spart Zeit)
TARGET_SYNC   = 1_000       # alle N Lernschritte Target-Netz nachziehen
EPS_START, EPS_END, EPS_STEPS = 1.0, 0.1, 500_000   # ε linear über so viele Trainingszüge ...
EPS_EVERY     = 10_000       # ... aber nur alle N Trainingszüge eine Stufe runter (Treppe)
CHECKPOINTS   = (1_000, 2_500, 5_000, 10_000)   # nach so vielen Spielen eine Kopie: checkpoints/agent_1000.pth


# =====================================================================
# Netz
# =====================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = FeatureEngineering.MAP_SIZE + FeatureEngineering.GLOBAL_SIZE
        layers = []
        for d in DILATIONS:
            layers += [nn.Conv2d(c, HIDDEN, 3, padding=d, dilation=d), nn.ReLU()]
            c = HIDDEN
        layers.append(nn.Conv2d(HIDDEN, N_ACTIONS + 1, 1))   # 1x1: Q pro Bewegung + Sap-Karte
        self.body = nn.Sequential(*layers)

    def forward(self, maps, glob):
        """maps (B, C, W, H), glob (B, G) -> Q (B, A, W, H)."""
        g = glob[:, :, None, None].expand(-1, -1, maps.shape[2], maps.shape[3])
        return self.body(torch.cat([maps, g], dim=1))


# =====================================================================
# Replay-Buffer: Ring aus festen numpy-Arrays, platzsparend
# =====================================================================
DTYPES = dict(maps=np.float16, next_maps=np.float16, pos=np.int8, next_pos=np.int8, actions=np.int8,
              sap_target=np.int8, sap_range=np.int8)


class ReplayBuffer:
    def __init__(self, size):
        self.size, self.n, self.i = size, 0, 0
        self.data = None                       # wird beim ersten push angelegt (dann sind die Formen bekannt)

    def push(self, **t):
        if self.data is None:
            self.data = {k: np.zeros((self.size,) + np.shape(v), dtype=DTYPES.get(k, np.asarray(v).dtype))
                         for k, v in t.items()}
        for k, v in t.items():
            self.data[k][self.i] = v
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        return {k: v[idx] for k, v in self.data.items()}


# =====================================================================
# Modell
# =====================================================================
class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps = 0                         # gesehene Trainings-Züge (für ε)
        self.games = 0                         # fertig trainierte Spiele (für Checkpoints)
        self.updates = 0                       # Lernschritte (für Target-Sync)
        self.loss_sum = self.q_sum = 0.0       # fürs Log: Summen über alle Lernschritte

    # ---------- Aktionswahl ----------
    def epsilon(self):
        f = min(self.steps // EPS_EVERY * EPS_EVERY / EPS_STEPS, 1.0)
        return EPS_START + f * (EPS_END - EPS_START)

    def choose(self, x, pos, alive, mask, sap_ok, sap_range, explore=True):
        """Aktionen für alle Units. x = (maps, globals), pos (N, 2), alive (N,), mask (N, A),
        sap_ok (N,) genug Energie, sap_range Reichweite -> (N, 3) mit [typ, dx, dy]."""
        actions = np.zeros((len(alive), 3), dtype=int)
        units = np.flatnonzero(alive)
        if len(units) == 0:
            return actions
        mask = _stay_allowed(mask)[units]

        with torch.no_grad():
            out = self.net(*self._tensors([x[0]], [x[1]]))[0].cpu().numpy() # (A + 1, W, H)
        q = out[:N_ACTIONS, pos[units, 0], pos[units, 1]].T                 # (K, A)
        q[~mask] = -np.inf                                                  # verboten -> nie wählen
        order = np.argsort(-q, axis=1)                                      # beste Aktion zuerst, verbotene hinten
        best = order[:, 0].copy()

        # Units auf demselben Feld haben dieselben Q-Werte -> würden ewig als Klumpen laufen.
        # Darum: k-te Unit auf einem Feld nimmt die k-tbeste erlaubte Aktion (mehr Units als Aktionen: von vorn).
        seen = {}
        for k, u in enumerate(units):
            t = tuple(pos[u])
            seen[t] = seen.get(t, -1) + 1
            best[k] = order[k, seen[t] % mask[k].sum()]

        # Sap: bestes Feld in Reichweite gegen die gewählte Bewegung
        sap_map, area = out[SAP], _sap_area(pos[units], sap_range, out.shape[1:])
        area[~np.asarray(sap_ok)[units]] = False
        for k, u in enumerate(units):
            if area[k].any():
                t = np.unravel_index(np.argmax(np.where(area[k], sap_map, -np.inf)), sap_map.shape)
                if sap_map[t] > q[k, best[k]]:
                    actions[u] = (SAP, t[0] - pos[u, 0], t[1] - pos[u, 1])

        if explore:                                                         # ε-greedy pro Unit
            for k in np.flatnonzero(np.random.rand(len(units)) < self.epsilon()):
                u, moves = units[k], np.flatnonzero(mask[k])
                # Sap zählt wie EINE Aktion, sonst wären fast alle Zufallszüge Saps (bis zu 225 Felder)
                if area[k].any() and np.random.randint(len(moves) + 1) == 0:
                    tiles = np.argwhere(area[k])
                    tx, ty = tiles[np.random.randint(len(tiles))]
                    actions[u] = (SAP, tx - pos[u, 0], ty - pos[u, 1])
                else:
                    actions[u] = 0
                    best[k] = np.random.choice(moves)

        move = actions[units, 0] != SAP
        actions[units[move], 0] = best[move]
        return actions

    # ---------- Training ----------
    def train_step(self, x, pos, alive, actions, reward, next_x, next_pos, next_alive, next_mask,
                   next_sap_ok, sap_range, done):
        """Übergang speichern und ab und zu einen Lernschritt machen."""
        if not alive.any():                       # z.B. Matchanfang: keine Unit hat gehandelt
            return
        self.buffer.push(maps=x[0], glob=x[1], pos=pos, alive=alive, actions=actions[:, 0],
                         sap_target=pos + actions[:, 1:], sap_range=sap_range,
                         reward=reward, next_maps=next_x[0], next_glob=next_x[1], next_pos=next_pos,
                         next_alive=next_alive, next_mask=_stay_allowed(next_mask),
                         next_sap_ok=next_sap_ok, done=bool(done))
        self.steps += 1
        if self.buffer.n >= BATCH_SIZE and self.steps % TRAIN_EVERY == 0:
            self._learn(self.buffer.sample(BATCH_SIZE))

    def _learn(self, b):
        bi, ui = np.nonzero(b["alive"])                                     # alle Units, die gehandelt haben
        px, py = b["pos"][bi, ui].T.astype(int)
        nx, ny = np.maximum(b["next_pos"][bi, ui], 0).T.astype(int)         # tote Units: -1 -> 0 (zählt eh nicht)
        a = b["actions"][bi, ui].astype(int)
        sap = a == SAP
        qx, qy = np.where(sap, b["sap_target"][bi, ui].T.astype(int), (px, py))   # Sap liest am Zielfeld
        nok = b["next_sap_ok"][bi, ui]
        nrange = b["sap_range"][bi].astype(int)
        r = torch.as_tensor(b["reward"][bi, ui], device=self.device)
        go_on = torch.as_tensor(b["next_alive"][bi, ui] & ~b["done"][bi], device=self.device).float()
        nmask = torch.as_tensor(b["next_mask"][bi, ui], device=self.device)
        bi, ui, qx, qy, nx, ny, a = (torch.as_tensor(v, device=self.device) for v in (bi, ui, qx, qy, nx, ny, a))

        q = self.net(*self._tensors(b["maps"], b["glob"]))[bi, a, qx, qy]  # Q(s, a) pro Unit

        with torch.no_grad():
            nmaps, nglob = self._tensors(b["next_maps"], b["next_glob"])
            online, tgt = self.net(nmaps, nglob), self.target(nmaps, nglob)
            q_next = online[bi, :N_ACTIONS, nx, ny]                         # (K, A) Online-Netz wählt ...
            q_next[~nmask] = -float("inf")
            best_move, a_next = q_next.max(dim=1)
            q_eval = tgt[bi, a_next, nx, ny]                                # ... Target-Netz bewertet

            # dasselbe für Sap: bestes erlaubtes Feld um next_pos
            area = _sap_area(np.stack([nx.cpu().numpy(), ny.cpu().numpy()], axis=1), nrange, online.shape[2:])
            area[~nok] = False
            area = torch.as_tensor(area, device=self.device)
            s_next = online[bi, SAP].masked_fill(~area, -float("inf")).flatten(1)
            best_sap, t_next = s_next.max(dim=1)
            H = online.shape[3]
            sap_eval = tgt[bi, SAP, t_next // H, t_next % H]
            q_eval = torch.where(best_sap > best_move, sap_eval, q_eval)
            y = r + GAMMA * go_on * q_eval

        loss = nn.functional.smooth_l1_loss(q, y)                           # Huber
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()

        self.loss_sum += loss.item()
        self.q_sum += q.mean().item()
        self.updates += 1
        if self.updates % TARGET_SYNC == 0:
            self.target.load_state_dict(self.net.state_dict())

    # ---------- Speichern ----------
    def save(self, path):
        """Netz + Zähler speichern. Bei einer Spielzahl aus CHECKPOINTS zusätzlich eine Kopie in checkpoints/."""
        state = {"net": self.net.state_dict(), "steps": self.steps, "games": self.games}
        torch.save(state, path)
        if self.games in CHECKPOINTS:
            folder = os.path.join(os.path.dirname(os.path.abspath(path)), "checkpoints")
            os.makedirs(folder, exist_ok=True)
            torch.save(state, os.path.join(folder, f"agent_{self.games}.pth"))

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        # Alte .pth ohne Sap-Karte: Bewegungs-Kanäle übernehmen, Sap-Kanal bleibt neu
        net = self.net.state_dict()
        for k, v in state["net"].items():
            if v.shape != net[k].shape:
                net[k][:v.shape[0]] = v
                state["net"][k] = net[k]
        self.net.load_state_dict(state["net"])
        self.target.load_state_dict(state["net"])
        self.steps = state["steps"]
        self.games = state.get("games", 0)        # alte .pth ohne Spielzähler: ab 0 zählen

    # ---------- Hilfe ----------
    def _tensors(self, maps, glob):
        """numpy (float16/32) -> float32-Tensoren auf dem Gerät."""
        return (torch.as_tensor(np.asarray(maps, dtype=np.float32), device=self.device),
                torch.as_tensor(np.asarray(glob, dtype=np.float32), device=self.device))


def _sap_area(pos, sap_range, size):
    """(K, W, H) bool: Felder im Quadrat um jede Position. pos (K, 2), sap_range Zahl oder (K,), size (W, H)."""
    pos, r = np.asarray(pos), np.asarray(sap_range)
    xs, ys = np.arange(size[0])[None, :], np.arange(size[1])[None, :]
    near_x = np.abs(xs - pos[:, :1]) <= (r[:, None] if r.ndim else r)
    near_y = np.abs(ys - pos[:, 1:]) <= (r[:, None] if r.ndim else r)
    return near_x[:, :, None] & near_y[:, None, :]


def _stay_allowed(mask):
    """STAY geht immer: falls eine Unit gar nichts darf, wird STAY erlaubt."""
    mask = np.array(mask, dtype=bool)
    mask[:, 0] |= ~mask.any(axis=1)
    return mask


# =====================================================================
# Ein Modell pro Pfad, solange Python läuft (über alle Spiele, beide Spieler)
# =====================================================================
_MODELS = {}


def get_model(path):
    if path not in _MODELS:
        model = Model()
        if path and os.path.isfile(path):
            model.load(path)
        _MODELS[path] = model
    return _MODELS[path]
