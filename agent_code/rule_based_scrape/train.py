"""
    This file implements the scraping into a large memory buffer.
"""

from typing import List
from agent_code.crazy_turtle.train import get_custom_events, reward_from_events
from agent_code.crazy_turtle.feature_extractor import state_to_features, get_state_tuple
from agent_code.crazy_turtle.util import *
from agent_code.rule_based_scrape.config import config
import json
from tqdm import tqdm
import sys


def setup_training(self):
    print("start training with parameters:")
    print(json.dumps(config, indent=2))


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is not None:
        # ------< get feature vectors >------
        f_state = state_to_features(old_game_state)
        f_state_new = state_to_features(new_game_state)
        action_index = action2index[self_action]
        # ------< handle events >------
        events += get_custom_events(f_state, action_index)
        r = reward_from_events(self, events)
        # print(r, events)
        # ------< handle model >------
        done = 0
        if len(self.agent.memory) != self.agent.memory.max_size:
            self.agent.memory.store(f_state, action_index, r, f_state_new, done)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    # ------< get feature vectors >------
    f_state = state_to_features(last_game_state)
    action_index = action2index[last_action]
    # ------< handle events >------
    events += get_custom_events(f_state, action_index)
    r = reward_from_events(self, events)
    # print(r, events)
    # ------< handle model >------
    done = 1
    if len(self.agent.memory) != self.agent.memory.max_size:
        self.agent.memory.store(f_state, action_index, r, f_state, done)
    else:
        tqdm.write("%%%%%%%%%%%%% REPLAY MEMORY FULL %%%%%%%%%%%%%")
        self.agent.memory.save_memories(config["MEMORY_PATH"])
        mem = self.agent.memory.get_all()
        print(mem["obs"])
        # old_mem = self.agent.memory.get_all()
        #
        # tqdm.write("%%%%%%%%%%%%% DEBUG: Reload memory %%%%%%%%%%%%%")
        # self.agent.memory.load_memories(config["MEMORY_PATH"])
        # mem = self.agent.memory.get_all()
        #
        # tqdm.write("%%%%%%%%%%%%% DEBUG: Reload memory - examine difference %%%%%%%%%%%%%")
        # print(old_mem["obs"] - mem["obs"])
        # print(old_mem["acts"] - mem["acts"])
        # print(old_mem["rews"] - mem["rews"])
        # print(old_mem["done"] - mem["done"])
        # print  (old_mem["next_obs"] - mem["next_obs"])

        sys.exit(0)

    self.agent.decay_epsilon()
    print(self.agent.epsilon)

    if last_game_state["round"] % config["TARGET_SAVE"] == 0:
        tqdm.write(
            f"EPSILON {self.agent.epsilon} | TRANSITIONS {len(self.agent.memory)} of {self.agent.memory.max_size} ({100 * len(self.agent.memory) / self.agent.memory.max_size}%)")

        tqdm.write("saving replay mem")
        self.agent.memory.save_memories(config["MEMORY_PATH"])
        # pprint.pprint((self.agent.memory.get_all()), width=2)
