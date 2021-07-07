import contextlib
import time
import traceback

import aicrowd_gym
import numpy as np
from multiprocessing import Pool
from nle import nethack as nh

from agent import Agent


class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.score = 0
        self.visualizer = None

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(nh.actions.ACTIONS.index(action))
        self.score += reward
        return obs, reward, done, info

    def debug_tiles(self, *args, **kwargs):
        return contextlib.suppress()

    def debug_log(self, *args, **kwargs):
        return contextlib.suppress()


def single_game(i):
    env = EnvWrapper(aicrowd_gym.make('NetHackChallenge-v0'))
    try:
        agent = Agent(env)
        agent.main()
    except BaseException as e:
        print(''.join(traceback.format_exception(None, e, e.__traceback__)))

    print(f'Run {i} finished with score {env.score}')

    return env.score

if __name__ == "__main__":
    NUM_ASSESSMENTS = 4096

    start_time = time.time()

    with Pool(4) as pool:
        scores = list(pool.map(single_game, range(NUM_ASSESSMENTS)))

    print('scores  :', scores)
    print('duration:', time.time() - start_time)
    print('len     :', len(scores))
    print('median  :', np.median(scores))
    print('mean    :', np.mean(scores))
