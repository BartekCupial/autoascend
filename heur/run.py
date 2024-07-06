import argparse
import ast
import contextlib
import ctypes
import os
import pickle
import random
import string
import sys
import threading
import time
import traceback
from multiprocessing import Pool

import gym
import numpy as np
from nle import nethack as nh

from heur.agent import Agent
from heur.character import Character
from heur.task_rewards_info import TaskRewardsInfoWrapper


def generate_random_string(length=15):
    # Define the character set
    characters = string.ascii_letters + string.digits

    # Generate the random string
    random_string = "".join(random.choice(characters) for _ in range(length))

    return random_string


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
        self.agent = None

    def reset(self):
        obs = self.env.reset()
        self.score = 0
        self.step_count = -1
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(nh.actions.ACTIONS.index(action))
        self.score += reward
        self.step_count += 1
        # if self.score >= 5650:
        #    if self.agent.character.role not in []:#self.agent.character.VALKYRIE]:
        #        for _ in range(5):
        #            action = nh.actions.ACTIONS.index(nh.actions.Command.ESC)
        #            obs, reward, done, info = self.env.step(action)
        #        for c in '#quit\ry':
        #            action = nh.actions.ACTIONS.index(ord(c))
        #            obs, reward, done, info = self.env.step(action)
        #        assert done
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
                if step_count != self.step_count or self._finished:
                    break
            else:
                for thread_id, thread in threading._active.items():
                    if thread is threading.main_thread():
                        break
                else:
                    assert 0, "main thread not found"
                out = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(thread_id), ctypes.py_object(AgentStepTimeout)
                )
                assert out == 1, out
                break

    def main(self, save_sokoban=False):
        timer_thread = threading.Thread(target=self._timer_thread)
        timer_thread.start()
        try:
            self.reset()
            self.agent = Agent(self, panic_on_errors=True, save_sokoban=save_sokoban)
            self.agent.main()
        finally:
            self._finished = True
            timer_thread.join()


def worker(args):
    from_, to_, flags = args

    gamesavedir = os.path.join(flags.gamesavedir, f"sokoban_{generate_random_string()}")
    orig_env = TaskRewardsInfoWrapper(
        gym.make(
            flags.game,
            save_ttyrec_every=1,
            savedir=flags.savedir,
            # gamesavedir=gamesavedir,
        )
    )

    scores = []
    extra_stats = []
    for i in range(from_, to_):
        orig_env.seed(i)
        env = EnvWrapper(orig_env)
        try:
            env.main(save_sokoban=flags.save_sokoban)
        except BaseException as e:
            print(
                "".join(traceback.format_exception(None, e, e.__traceback__)),
                file=sys.stderr,
            )

        print(f"Run {i} finished with score {env.score}", file=sys.stderr)

        scores.append(env.score)
        extra_stats.append(env.env.last_info)

    with open(f'data_{from_}.pkl', 'wb') as file:
        pickle.dump(extra_stats, file)

    orig_env.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--num_assessments", type=int)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--gamesavedir", type=str)
    parser.add_argument("--save_sokoban", type=ast.literal_eval, default=False)
    flags = parser.parse_args()

    start_time = time.time()

    with Pool(flags.num_threads) as pool:
        scores = list(
            pool.map(
                worker,
                [
                    (
                        i * flags.num_assessments // flags.num_threads,
                        (i + 1) * flags.num_assessments // flags.num_threads,
                        flags,
                    )
                    for i in range(flags.num_threads)
                ],
            )
        )
    scores = [s for ss in scores for s in ss]

    print("scores  :", scores, file=sys.stderr)
    print("duration:", time.time() - start_time, file=sys.stderr)
    print("len     :", len(scores), file=sys.stderr)
    print("median  :", np.median(scores), file=sys.stderr)
    print("mean    :", np.mean(scores), file=sys.stderr)
