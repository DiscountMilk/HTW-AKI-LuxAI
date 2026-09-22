# LuxAI

A small project for running Lux AI agents against each other, optionally with training and replay generation.

## Useful LuxAI links

- LuxAI Season 3: https://github.com/Lux-AI-Challenge/Lux-Design-S3
- Rules: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md
- Python Getting Started: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/python/README.md
- Actions and observations: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/README.md

## Project structure

- `main.py` - simple entry point for starting matches
- `lux/runner.py` - contains the central `run_agents(...)` function (no need to change anything)
- `agents/` - agent implementations
- `Paths.csv` - mapping of Moodle groups to import and load paths
- `replays/` - generated replay files
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- installed packages from `requirements.txt`

## Installation

Run this in the project directory:

```powershell
py -3 -m pip install -r requirements.txt
```

## Important course-specific change

Because this course setup plays only **3 matches per episode**, students should update the parameter below in their local virtual environment file:

- file: `.venv/luxai_s3/params.py`

Change this:

```python
match_count_per_episode: int = 5
```

To this:

```python
match_count_per_episode: int = 3
```

If your virtual environment is located somewhere else, make the same change in that environment's `luxai_s3/params.py` file.

## Running the project

The default entry point is `main.py`.

```powershell
py -3 main.py
```

At the moment, `main.py` starts the following setup:

- group `1` vs group `3`
- `train=False`
- `games_to_play=1`
- `save_frequency=2`

## `run_agents(...)`

The main function is located in `lux/runner.py`:

```text
run_agents(moodle_group_player_0, moodle_group_player_1, games_to_play=10, train=True, save_frequency=500)
```

### Parameters

- `moodle_group_player_0` - group ID for player 0
- `moodle_group_player_1` - group ID for player 1
- `games_to_play` - number of games to run
- `train` - if `True`, learning updates are executed; if `False`, replays are generated
- `save_frequency` - saves agent state every `N` games if the agent implements `save_agent()`

## Replays

If `train=False`, replay files are written to `replays/<group0>_vs_<group1>/`.

You can visualize the replays here:

- https://s3vis.lux-ai.org/

## Agent configuration

Agent selection is controlled through `Paths.csv`.

The file is expected to contain at least these columns:

- `Moodle Gruppe`
- `Import_Path`
- `Load_Path_P0` (if you're playing as player 0 and want to load agent trained as player 0)
- `Load_Path_P1` (if you're playing as player 1 and want to load agent trained as player 1)

`Import_Path` can be either:

- a Python import path like `package.module.ClassName`
- or a file path in the format `file.py:ClassName`

## Agent methods

Depending on the agent, the runner uses these methods if they exist:

- `act(step, obs)` – Returns the action for the current game step. `step` is the current time step (integer) within an episode, and `obs` is the observation dictionary provided by the environment for the agent's player perspective.

- `update(step, obs, reward, action, terminated, next_obs)` – Called after each step to allow the agent to learn from experience (TD-Learning style - in case of Monte Carlo full episodes must be stored within agent script and in agent.update() the actual updates are only performed if episode terminated). `step` is the current time step, `obs` is the observation before the action, `reward` is the scalar reward received, `action` is the action that was taken, `terminated` is a boolean indicating whether the episode has ended, and `next_obs` is the observation after the action was applied.

- `save_agent()` – Persists the agent's current state (e.g. model weights, replay buffer) to disk. Called automatically by the runner every `save_frequency` games if this method is present.

- `set_to_eval_mode()` – Switches the agent into evaluation mode, disabling any training-specific behaviour such as exploration noise or gradient computation. Called by the runner when `train=False`.

- `set_env_cfg(env_cfg)` – Passes the environment configuration object to the agent before the first game starts. Useful for reading environment constants such as map size, unit parameters, or match length.
