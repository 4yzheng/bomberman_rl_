"""
    This file implements the rule-based scraping agent.
"""

from agent_code.crazy_turtle.model import ReplayBuffer
from agent_code.rule_based_scrape.config import config
from tqdm import tqdm


class DeterministicAgent:
    """
        Implementation of very simple rule-based scraping agent.
    """
    def __init__(self, epsilon=1.0, size=60000):
        tqdm.write(" -- SCRAPING BOMBERMAN DATA --")
        tqdm.write(f"REPLAY MEMORY HAS {size} CAPACITY")
        self.memory = ReplayBuffer(size)
        self.epsilon = epsilon
        self.epsilon_decay = (self.epsilon - config["EPSILON_MIN"]) / config["EPSILON_MIN_EPISODE"]

    def decay_epsilon(self):
        """
            Decrease the epsilon in the epsilon-greedy action selection.
        """
        if self.epsilon >= config["EPSILON_MIN"]:
            self.epsilon -= self.epsilon_decay