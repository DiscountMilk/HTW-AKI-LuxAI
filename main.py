from lux.runner import run_agents

if __name__ == "__main__":
    run_agents(
        7,
        4,
        train=False, # Set to True to enable learning updates (if your agent implements update()). Set to False to disable learning and just generate replays.
        games_to_play=1,
        save_frequency=50, # Save agent state every N games (if your agent implements save_agent() and load_agent() methods
    )

    # Visualize game replays at https://s3vis.lux-ai.org/
