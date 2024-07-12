import argparse
import ast
import sys
import time
import traceback
from pathlib import Path

import gym
import pandas as pd

from heur.run import EnvWrapper
from nle_utils.collections import concat_dicts
from nle_utils.parallel_utils import Result, map_parallel
from nle_utils.utils import str2bool
from nle_utils.wrappers import (
    FinalStatsWrapper,
    LastInfo,
    NLEDemo,
    RenderTiles,
    TaskRewardsInfoWrapper,
    TtyrecInfoWrapper,
)


def worker(seed, game: str, savedir: str, gamesavedir: str, save_video: bool, save_demo: bool):
    orig_env = gym.make(
        game,
        save_ttyrec_every=1,
        savedir=savedir,
    )
    orig_env = TaskRewardsInfoWrapper(orig_env, done_only=False)
    orig_env = FinalStatsWrapper(orig_env, done_only=False)
    orig_env = TtyrecInfoWrapper(orig_env, done_only=False)
    orig_env = LastInfo(orig_env)
    if save_video:
        orig_env = RenderTiles(orig_env, output_dir=gamesavedir)
    if save_demo:
        orig_env = NLEDemo(orig_env, gamesavedir)
    if seed is not None:
        orig_env.seed(seed)

    env = EnvWrapper(orig_env)

    try:
        env.main()
        error_message = None
    except BaseException as e:
        error_message = "".join(traceback.format_exception(None, e, e.__traceback__))

    if save_demo:
        orig_env.save_to_file()

    message = f"{orig_env.get_seeds()}"
    result = orig_env.last_info.get("episode_extra_stats", {})

    orig_env.close()
    return Result(result, description=message, log_msg=error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_assessments", type=int)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--gamesavedir", type=str)
    parser.add_argument("--experiment_id", type=int, default=0)
    parser.add_argument("--seed", type=str2bool, default=True)
    parser.add_argument("--save_demo", type=ast.literal_eval, default=False)
    parser.add_argument("--save_video", type=ast.literal_eval, default=False)
    parser.add_argument("--n_jobs", type=int, default=8)

    flags = parser.parse_args()
    print(flags)

    start_time = time.time()

    seeds = list(range(flags.num_assessments)) if flags.seed else [None] * flags.num_assessments
    total = len(seeds)
    data = map_parallel(
        function=worker,
        iterable=seeds,
        function_args=(flags.game, flags.savedir, flags.gamesavedir, flags.save_video, flags.save_demo),
        n_jobs=flags.n_jobs,
    )

    df = pd.DataFrame.from_dict(concat_dicts(data), orient="index").T
    csv_output_path = Path(flags.gamesavedir) / "stats.csv"
    csv_output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_output_path)
    scores = df["score"]
    agg = scores.agg(["mean", "max", "median"])

    print("scores  :", list(scores), file=sys.stderr)
    print("duration:", time.time() - start_time, file=sys.stderr)
    print("len     :", len(df), file=sys.stderr)
    print("median  :", agg.loc["median"], file=sys.stderr)
    print("mean    :", agg.loc["mean"], file=sys.stderr)
