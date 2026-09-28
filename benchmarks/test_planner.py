"""Regression tests for engine-specific bomb timing and the safety policy."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
import numpy as np
from agent_code.planner_agent.callbacks import Forecast, act, blast, setup
from environment import GenericWorld
from items import Bomb
import settings as s


def state():
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, self=('test', 0, True, (3, 3)), others=[], bombs=[],
                coins=[], explosion_map=np.zeros_like(field), step=1, round=1)


class PlannerTests(unittest.TestCase):
    def test_blast_matches_engine_through_crates_and_stops_at_walls(self):
        game = state()
        game['field'][4, 3] = 1
        game['field'][3, 5] = -1
        bomb = Bomb((3, 3), None, 0, s.BOMB_POWER, None)
        self.assertEqual(blast(game['field'], (3, 3)), set(bomb.get_blast_coords(game['field'])))
        self.assertIn((5, 3), blast(game['field'], (3, 3)))
        self.assertNotIn((3, 6), blast(game['field'], (3, 3)))

    def test_forecast_matches_actual_engine_explosions_and_crate_clearance(self):
        game = state()
        game['bombs'] = [((3, 3), 0), ((5, 3), 3)]
        game['field'][4, 3] = 1
        prediction = Forecast(game)
        owner = SimpleNamespace(add_event=lambda event: None, bombs_left=False, name='test')
        world = SimpleNamespace(arena=game['field'].copy(), explosions=[], coins=[],
            logger=SimpleNamespace(info=lambda *args: None),
            bombs=[Bomb(p, owner, timer, s.BOMB_POWER, None) for p, timer in game['bombs']])
        for t in range(1, prediction.horizon + 1):
            # Movement precedes explosions. A bomb hit by another blast keeps its timer.
            self.assertEqual(bool(prediction.open[t][4, 3]), world.arena[4, 3] == 0)
            GenericWorld.update_explosions(world)
            GenericWorld.update_bombs(world)
            danger = np.zeros_like(game['field'], dtype=bool)
            for explosion in world.explosions:
                if explosion.is_dangerous():
                    for p in explosion.blast_coords:
                        danger[p] = True
            np.testing.assert_array_equal(prediction.danger[t], danger)

    def test_current_explosion_expires_before_next_movement_evaluation(self):
        game = state()
        game['explosion_map'][3, 3] = 1
        forecast = Forecast(game)
        self.assertTrue(forecast.danger[1][3, 3])
        self.assertFalse(forecast.danger[2][3, 3])

    def test_refuses_bomb_in_sealed_corridor(self):
        game = state()
        game['field'][:] = -1
        game['field'][2:5, 3] = 0
        game['field'][5, 3] = 1
        self.assertFalse(Forecast(game, plant=True).routes(plant=True))
        agent = SimpleNamespace()
        setup(agent)
        self.assertNotEqual(act(agent, game), 'BOMB')

    def test_escapes_timer_zero_bomb_and_collects_adjacent_coin(self):
        game = state()
        game['bombs'] = [((2, 3), 0)]
        agent = SimpleNamespace()
        setup(agent)
        self.assertIn(act(agent, game), ['UP', 'DOWN'])
        game['bombs'] = []
        game['coins'] = [(4, 3)]
        self.assertEqual(act(agent, game), 'RIGHT')

    def test_leaves_blast_lane_early_even_when_a_coin_is_reachable(self):
        game = state()
        game['bombs'] = [((3, 4), 3)]
        game['coins'] = [(3, 2)]
        agent = SimpleNamespace()
        setup(agent)
        self.assertIn(act(agent, game), ['LEFT', 'RIGHT'])


if __name__ == '__main__':
    unittest.main()
