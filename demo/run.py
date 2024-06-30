import argparse
import sys
import time
import traceback
import os

from multiprocessing import Pool

import gym
import numpy as np

from demo.wrappers import EnvWrapper, NLEDemo


def worker(args):
    from_, to_, flags = args
    orig_env = gym.make(
        flags.game,
        save_ttyrec_every=1,
        savedir=flags.savedir,
    )

    scores = []
    for i in range(from_, to_):
        env = EnvWrapper(NLEDemo(orig_env, flags.gamesavedir))
        try:
            env.main()
        except BaseException as e:
            print(
                "".join(traceback.format_exception(None, e, e.__traceback__)),
                file=sys.stderr,
            )

        print(f"Run {i} finished with score {env.score}", file=sys.stderr)

        env.save_to_file()

        scores.append(env.score)
    orig_env.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--num_assessments", type=int)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--gamesavedir", type=str)
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
