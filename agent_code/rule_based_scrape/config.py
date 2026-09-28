"""
    This file stores the parameters for the rule-based scraping agent.
"""

# --------------< Parameters >--------------<
config = {
    "STATE_DIM": 5,
    "ACTION_DIM": 6,
    "REPLAY_BUFFER_SIZE": 7000,  # 10000000,
    "TARGET_SAVE": 10,
    "MEMORY_PATH": "",
    "EPSILON_MIN": 0.1,
    "EPSILON_MIN_EPISODE": 3,  # 500000,
}

mode = "classic"
enemies = "alone"

config[
    "MEMORY_PATH"] = f"""memory_eps=[1,{config["EPSILON_MIN"]}][0,{config["EPSILON_MIN_EPISODE"]}]_n={config["REPLAY_BUFFER_SIZE"]}_{enemies}_{mode}.pt"""
