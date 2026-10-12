from lux.runner import run_agents

if __name__ == "__main__":
    run_agents(
        7,
        2,
        train=False, # Set to True to enable learning updates (if your agent implements update()). Set to False to disable learning and just generate replays.
        games_to_play=10,
        matches_per_game=3, # Matches pro Spiel (offizielle Regeln: 5)
        save_frequency=2, # Save agent state every N games (if your agent implements save_agent() and load_agent() methods
    )

    # Visualize game replays at https://s3vis.lux-ai.org/
