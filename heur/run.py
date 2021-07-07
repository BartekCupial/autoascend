import contextlib
import time
import traceback

from multiprocessing import Pool
import aicrowd_gym
import numpy as np

from agent import Agent


class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.score = 0

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(self.env._actions.index(action))
        self.score += reward
        return obs, reward, done, info

    def debug_tiles(self, *args, **kwargs):
        return contextlib.suppress()

    def debug_log(self, *args, **kwargs):
        return contextlib.suppress()


def single_game(_=None):
    env = EnvWrapper(aicrowd_gym.make('NetHackChallenge-v0'))
    try:
        agent = Agent(env)
        agent.main()
    except BaseException as e:
        print(''.join(traceback.format_exception(None, e, e.__traceback__)))

    return env.score

if __name__ == "__main__":
    NUM_ASSESSMENTS = 4096

    start_time = time.time()

    with Pool(4) as pool:
        scores = list(pool.map(single_game, [None] * NUM_ASSESSMENTS))

    print('duration:', time.time() - start_time)
    print('len     :', len(scores))
    print('median  :', np.median(scores))
    print('mean    :', np.mean(scores))
