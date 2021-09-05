import contextlib
import ctypes
import sys
import threading
import time
import traceback

import aicrowd_gym
import numpy as np
from multiprocessing import Pool
from nle import nethack as nh

from agent import Agent


class AgentStepTimeout(KeyboardInterrupt):
    # it inheirits from KeyboardInterrupt because agent never catches it
    pass

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.score = 0
        self.step_count = -1
        self.visualizer = None
        self._finished = False

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        self.step_count = -1
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(nh.actions.ACTIONS.index(action))
        self.score += reward
        self.step_count += 1
        return obs, reward, done, info

    def debug_tiles(self, *args, **kwargs):
        return contextlib.suppress()

    def debug_log(self, *args, **kwargs):
        return contextlib.suppress()

    def _timer_thread(self):
        while not self._finished:
            step_count = self.step_count
            for _ in range(20):
                time.sleep(0.25)
                if step_count != self.step_count:
                    break
            else:
                for thread_id, thread in threading._active.items():
                    if thread is threading.main_thread():
                        break
                else:
                    assert 0, 'main thread not found'
                out = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id),
                                                                 ctypes.py_object(AgentStepTimeout))
                assert out == 1, out
                break

    def main(self):
        timer_thread = threading.Thread(target=self._timer_thread)
        timer_thread.start()
        try:
            self.reset()
            agent = Agent(self)
            agent.main()
        finally:
            self._finished = True
            timer_thread.join()


def single_game(i):
    orig_env = aicrowd_gym.make('NetHackChallenge-v0')
    if orig_env is None:
        print(f'Run {i} not available')
        return

    try:
        env = EnvWrapper(orig_env)
        try:
            env.main()
        except BaseException as e:
            print(''.join(traceback.format_exception(None, e, e.__traceback__)))
    finally:
        orig_env.close()

    print(f'Run {i} finished with score {env.score}')

    return env.score

if __name__ == "__main__":
    NUM_ASSESSMENTS = int(sys.argv[1])
    NUM_THREADS = int(sys.argv[2])

    start_time = time.time()

    with Pool(NUM_THREADS) as pool:
        scores = list(pool.map(single_game, range(NUM_ASSESSMENTS)))

    print('scores  :', scores)
    print('duration:', time.time() - start_time)
    print('len     :', len(scores))
    print('median  :', np.median(scores))
    print('mean    :', np.mean(scores))
