import os
import contextlib
import ctypes
import sys
import threading
import time
import traceback
import jsonlines
import tempfile
from pathlib import Path

import numpy as np
from multiprocessing import Pool
from nle import nethack as nh
from nle.nethack.actions import ACTIONS
from nle_language_wrapper import NLELanguageWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv

import gym

from heur.agent import Agent
from heur.character import Character
from heur.action_textmap import nle_action_textmap

NH_ACTION_STR_TO_IDX = {str(ACTIONS[i]): i for i in range(len(ACTIONS))}
NH_ACTION_IDX_TO_STR = {v: k for (k, v) in NH_ACTION_STR_TO_IDX.items()}


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
        #if self.score >= 5650:
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

    def get_data(self):
        return self.agent._data

    def get_summary(self):
        return {
            "score": self.score,
            # "steps": self.env._steps,
            "turns": self.agent.blstats.time,
            "level_num": len(self.agent.levels),
            "experience_level": self.agent.blstats.experience_level,
            "milestone": self.agent.global_logic.milestone,
            "panic_num": len(self.agent.all_panics),
            "character": str(self.agent.character).split()[0],
            # "end_reason": self.end_reason,
            # "seed": self.env.get_seeds(),
            **self.agent.stats_logger.get_stats_dict(),
        }

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
            self.agent = Agent(self, panic_on_errors=True)
            self.agent.main()
        finally:
            self._finished = True
            timer_thread.join()


def worker(args):
    from_, to_, savedir, textsavedir = args
    
    os.makedirs(savedir, exist_ok=True)
    savedir = tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S_"), dir=savedir)
    textsavedir = os.path.join(textsavedir, Path(savedir).name)
    os.makedirs(textsavedir, exist_ok=True)
    
    orig_env = gym.make('NetHackChallenge-v0', save_ttyrec_every=1, savedir=savedir)

    nle_language = NLELanguageObsv()

    scores = []
    for i in range(from_, to_):
        env = EnvWrapper(orig_env)
        try:
            env.main()
        except BaseException as e:
            print(''.join(traceback.format_exception(None, e, e.__traceback__)), file=sys.stderr)

        summary = env.get_summary()

        json_safe_summary = {}
        for key, val in summary.items():
            if (
                isinstance(val, int)
                or isinstance(val, str)
                or isinstance(val, float)
                or isinstance(val, tuple)
            ):
                json_safe_summary[key] = val
            else:
                json_safe_summary[key] = val.item()

        text_data = [json_safe_summary]

        data = env.get_data()

        for ts in range(len(data)):
            datum = data[ts]

            txt_blstats = nle_language.text_blstats(datum["blstats"]).decode(
                "latin-1"
            )
            txt_glyphs = nle_language.text_glyphs(
                datum["glyphs"], datum["blstats"]
            ).decode("latin-1")
            txt_message = nle_language.text_message(datum["tty_chars"]).decode(
                "latin-1"
            )
            txt_inventory = nle_language.text_inventory(
                datum["inv_strs"], datum["inv_letters"]
            ).decode("latin-1")
            txt_cursor = (
                nle_language.text_cursor(
                    datum["glyphs"], datum["blstats"], datum["tty_chars"]
                ).decode("latin-1"),
            )
            if ts < len(data) - 1:
                txt_action = nle_action_textmap[data[ts + 1]["action"]]
            else:
                txt_action = "esc"

            text_datum = {
                "txt_blstats": txt_blstats,
                "txt_glyphs": txt_glyphs,
                "txt_message": txt_message,
                "txt_inventory": txt_inventory,
                "txt_cursor": txt_cursor,
                "txt_action": txt_action,
            }

            text_data += [text_datum]

        fn = f"{Path(env.env.nethack._ttyrec).stem}.jsonl"
        with jsonlines.open(os.path.join(textsavedir, fn), "w") as writer:
            writer.write_all(text_data)

        scores.append(env.score)

        print(f'Run {i} finished with score {env.score}', file=sys.stderr)

        # avoid memory leaks
        del env

    orig_env.close()
    return scores

if __name__ == "__main__":
    NUM_ASSESSMENTS = int(sys.argv[1])
    NUM_THREADS = int(sys.argv[2])

    start_time = time.time()

    with Pool(NUM_THREADS) as pool:
        scores = list(
            pool.map(worker, [
                (
                    i * NUM_ASSESSMENTS // NUM_THREADS,
                    (i + 1) * NUM_ASSESSMENTS // NUM_THREADS,
                    sys.argv[3],
                    sys.argv[4],
                )
                for i in range(NUM_THREADS)
            ]
        ))
    scores = [s for ss in scores for s in ss]

    print('scores  :', scores, file=sys.stderr)
    print('duration:', time.time() - start_time, file=sys.stderr)
    print('len     :', len(scores), file=sys.stderr)
    print('median  :', np.median(scores), file=sys.stderr)
    print('mean    :', np.mean(scores), file=sys.stderr)
