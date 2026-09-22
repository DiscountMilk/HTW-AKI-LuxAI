import glob
import importlib.util
import inspect
import json
import os
import sys
from importlib import import_module

import numpy as np
import pandas
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode


# Make the local `agents/` folder importable as top-level packages (so 'team_01' works when it's under agents/)
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_agents_dir = os.path.join(_base_dir, "agents")
if os.path.isdir(_agents_dir) and _agents_dir not in sys.path:
    # Put agents dir first so local teams shadow any global packages with same names
    sys.path.insert(0, _agents_dir)


def load_class_from_path(path: str):
    """
    Lädt eine Klasse aus einem dotted module path (module.Class) oder aus einem Dateipfad
    mit der Notation file_path:ClassName (z. B. C:/.../team_01.py:AgentClass).
    Falls das angegebene Modul nicht gefunden wird, versucht die Funktion automatisch,
    das Modul als Submodul des lokalen `agents`-Pakets zu importieren (z. B. `agents.team_01...`).
    """
    # Datei-Pfad-Notation: "C:/.../file.py:ClassName"
    if ":" in path and (os.path.exists(path.split(":", 1)[0]) or path.split(":", 1)[0].endswith(".py")):
        file_part, class_name = path.split(":", 1)
        file_path = os.path.abspath(file_part)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: `{file_path}`")
        module_name = f"loaded_module_{abs(hash(file_path))}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Kann Modul-Spec nicht erstellen für `{file_path}`")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"Klasse `{class_name}` nicht in Datei `{file_path}` gefunden") from exc

    # Dotted module path: package.module.ClassName
    try:
        module_path, class_name = path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(f"Ungültiger Pfad: `{path}`. Erwartet `module.ClassName` oder `file.py:ClassName`") from exc

    # Versuch 1: Direkt importieren
    tried = []
    try:
        module = import_module(module_path)
        return getattr(module, class_name)
    except ModuleNotFoundError as exc:
        tried.append(module_path)
        # Falls Modul nicht gefunden, versuchen wir ein Fallback mit dem lokalen 'agents' Paket
        agents_dir = os.path.join(_base_dir, "agents")
        first_segment = module_path.split(".", 1)[0]
        if os.path.isdir(agents_dir) and os.path.exists(os.path.join(agents_dir, first_segment)):
            alt_module = f"{first_segment}"
            tried.append(alt_module)
            try:
                module = import_module(alt_module)
                return getattr(module, class_name)
            except Exception:
                pass
        # Fallback gescheitert – klare Fehlermeldung mit Hinweisen
        raise ModuleNotFoundError(
            f"Kein Modul namens `{module_path}` gefunden (versucht: {tried}).\n"
            f"Stelle sicher, dass der Pfad in der CSV korrekt ist oder dass das Arbeitsverzeichnis so gesetzt ist,\n"
            f"dass Python das Package finden kann. Wenn das Paket in `agents/` liegt, benutze z.B.\n"
            f"`agents.{module_path}` als Import-Pfad oder setzte das Arbeitsverzeichnis auf das Projekt-Root."
        ) from exc
    except AttributeError as exc:
        raise ImportError(f"Klasse `{class_name}` nicht im Modul `{module_path}` gefunden") from exc


def inject_names(path, name_p0, name_p1):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    md = j.setdefault("metadata", {})
    players = md.setdefault("players", {})
    players["player_0"] = name_p0
    players["player_1"] = name_p1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(j, f)


def _get_path_value(paths_df, moodle_group, column_name):
    matches = paths_df.loc[paths_df["Moodle Gruppe"] == moodle_group, column_name].tolist()
    if not matches:
        raise ValueError(f"Keine Zuordnung für Moodle Gruppe `{moodle_group}` in Spalte `{column_name}` gefunden.")
    return matches[0]


def run_agents(moodle_group_player_0, moodle_group_player_1, games_to_play=10, train=True, save_frequency=500):
    paths = pandas.read_csv(os.path.join(_base_dir, "Paths.csv"), encoding="cp1252", sep=";")

    import_path_player_0 = _get_path_value(paths, moodle_group_player_0, "Import_Path")
    load_path_player_0 = _get_path_value(paths, moodle_group_player_0, "Load_Path_P0")

    import_path_player_1 = _get_path_value(paths, moodle_group_player_1, "Import_Path")
    load_path_player_1 = _get_path_value(paths, moodle_group_player_1, "Load_Path_P1")

    agent_class_player_0 = load_class_from_path(import_path_player_0)
    agent_class_player_1 = load_class_from_path(import_path_player_1)

    replay_dir = os.path.join(_base_dir, f"replays/{moodle_group_player_0}_vs_{moodle_group_player_1}")
    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)

    if train:
        env = LuxAIS3GymEnv(numpy_output=True)
    else:
        # Set up Lux AI environment with recording wrapper
        env = RecordEpisode(
            LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_dir
        )

    # Main loop to play multiple games
    for i in range(games_to_play):

        # Instantiate agents
        sig_0 = inspect.signature(agent_class_player_0.__init__)
        if "agent_path" not in sig_0.parameters:
            player_0 = agent_class_player_0("player_0")
        else:
            player_0 = agent_class_player_0("player_0", agent_path=load_path_player_0)
        sig_1 = inspect.signature(agent_class_player_1.__init__)
        if "agent_path" not in sig_1.parameters:
            player_1 = agent_class_player_1("player_1")
        else:
            player_1 = agent_class_player_1("player_1", agent_path=load_path_player_1)

        # Set agents to eval mode if not training (e.g. set epsilon to 0 for epsilon-greedy policies, turn off dropout, etc.).
        # Requires agent to implement set_to_eval_mode() method.
        if not train:
            for agent in [player_0, player_1]:
                if hasattr(agent, "set_to_eval_mode"):
                    agent.set_to_eval_mode()


        # Reset environment and get initial observations
        rng = np.random.default_rng()
        seed_val = int(rng.integers(0, 1_000_000))
        print(f"Seed: {seed_val}")
        obs, info = env.reset(seed=seed_val)

        # Provide environment configuration to agents.
        # Only contains observable game parameters such as map size, max number of steps, etc.
        env_cfg = info["params"]
        if hasattr(player_0, "set_env_cfg"):
            player_0.set_env_cfg(env_cfg)
        if hasattr(player_1, "set_env_cfg"):
            player_1.set_env_cfg(env_cfg)

        # Save trained agent every n-th game.
        # Requires agent to implement save_agent() method.
        if train and i % save_frequency == 0:
            for agent in [player_0, player_1]:
                if hasattr(agent, "save_agent"):
                    agent.save_agent()

        # Main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:
            # Each agent takes an action
            # Requires agent to implement act() method.
            actions = {}
            for agent in [player_0, player_1]:
                if hasattr(agent, "act"):
                    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            # info["state"] contains unobservable game data that agents can't see

            # Check for end of game
            if isinstance(terminated, dict) and isinstance(truncated, dict):
                dones = {k: bool(terminated.get(k, False) or truncated.get(k, False)) for k in set(terminated) | set(truncated)}
            else:
                done_bool = bool(terminated) or bool(truncated)
                dones = {"player_0": done_bool, "player_1": done_bool}

            if dones.get("player_0", False) or dones.get("player_1", False):
                game_done = True

            # Let agents learn from this experience
            # Requires agent to implement update() method.
            if train:
                for agent in [player_0, player_1]:
                    if hasattr(agent, "update"):
                        agent_reward = reward.get(agent.player, 0.0) if isinstance(reward, dict) else reward
                        agent.update(
                            step=step,
                            obs=obs[agent.player],
                            reward=agent_reward,
                            action=actions[agent.player],
                            terminated=dones.get(agent.player, False),
                            next_obs=next_obs[agent.player],
                        )

            # Move to next step
            step += 1
            obs = next_obs

        if not train:
            # Stamp names into the replay JSON so the viewer shows them
            try:
                latest = max(glob.glob(os.path.join(replay_dir, "*.json")), key=os.path.getmtime)
                inject_names(latest, player_0.name, player_1.name)
            except ValueError:
                pass

    env.close()

    if not train:
        # Stamp names into the replay JSON so the viewer shows them
        try:
            latest = max(glob.glob(os.path.join(replay_dir, "*.json")), key=os.path.getmtime)
            inject_names(latest, player_0.name, player_1.name)
        except ValueError:
            pass

