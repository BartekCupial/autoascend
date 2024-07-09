import argparse
import ast
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import gym
import numpy as np
import pandas as pd

from demo.utils.collections import concat_dicts
from demo.utils.wrappers import EnvWrapper, FinalStatsWrapper, LastInfo, NLEDemo, RenderTiles, TaskRewardsInfoWrapper


def worker(args):
    from_, to_, flags = args
    orig_env = gym.make(
        flags.game,
        save_ttyrec_every=1,
        savedir=flags.savedir,
    )
    orig_env = TaskRewardsInfoWrapper(orig_env, done_only=False)
    orig_env = FinalStatsWrapper(orig_env, done_only=False)
    orig_env = LastInfo(orig_env)
    if flags.save_video:
        orig_env = RenderTiles(orig_env, output_path=flags.gamesavedir)
    if flags.save_demo:
        orig_env = NLEDemo(orig_env, flags.gamesavedir)

    data = []
    for i in range(from_, to_):
        env = EnvWrapper(orig_env)
        try:
            env.seed(i)
            env.main()
        except BaseException as e:
            print(
                "".join(traceback.format_exception(None, e, e.__traceback__)),
                file=sys.stderr,
            )

        print(f"Run {i} finished with score {env.score}", file=sys.stderr)

        if flags.save_demo:
            env.save_to_file()

        data.append(env.last_info.get("episode_extra_stats", {}))
    env.close()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int)
    parser.add_argument("--num_assessments", type=int)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--gamesavedir", type=str)
    parser.add_argument("--save_demo", type=ast.literal_eval, default=False)
    parser.add_argument("--save_video", type=ast.literal_eval, default=False)
    flags = parser.parse_args()

    start_time = time.time()

    with Pool(flags.num_threads) as pool:
        data = list(
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
    data = [s for ss in data for s in ss]
    df = pd.DataFrame(concat_dicts(data))
    agg = df.agg(["mean", "max", "median"])
    df.to_csv(Path(flags.gamesavedir) / "stats.csv")

    print("scores  :", list(df["score"]), file=sys.stderr)
    print("duration:", time.time() - start_time, file=sys.stderr)
    print("len     :", len(df), file=sys.stderr)
    print("median  :", agg["score"].loc["median"], file=sys.stderr)
    print("mean    :", agg["score"].loc["mean"], file=sys.stderr)
