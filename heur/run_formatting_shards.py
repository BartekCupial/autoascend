import jsonlines
import os
import sys
import traceback

from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from heur.action_textmap import (
    nle_comp_preqs,
    nle_obs_preqs,
    special_tokens_interaction_history,
)
from heur.instruction_encode_templates import encode_instruction_example


def form_prompt(data, obs_preqs):
    return "\n".join(
        [
            "%s[\n%s\n]" % (obs_preqs[key], data[key])
            for key in (
                "txt_blstats",
                "txt_glyphs",
                "txt_message",
                "txt_inventory",
                "txt_cursor",
            )
        ]
    )


def faster_load_and_process_chunks(
    input_dir,
    output_dir,
    seq,
    nsamples,
    observation_tok=special_tokens_interaction_history["observation"],
    obs_preqs=nle_obs_preqs,
    comp_preqs=nle_comp_preqs,
    instruction="You are an agent playing NetHack. Predict the next actions.",
):
    def process_helper_raw_observation(data):
        seq = len(data)
            
        query = form_prompt(data[0], obs_preqs)
        prompts = []
        actions = []
        for i in range(seq):
            if i < seq - 1:
                prompts += [form_prompt(data[i + 1], obs_preqs=obs_preqs)]
            actions += [data[i]["txt_action"]]

        obs = []
        for i in range(seq - 1):
            obs += [prompts[i]]

        completion = ""
        for j in range(seq):
            completion += "\n%s%s" % (
                comp_preqs["action"],
                actions[j],
            )
            if j < seq - 1:
                completion += "\n%s\n%s" % (observation_tok, obs[j])

        return encode_instruction_example(
            instruction,
            query,
            completion,
            random_template=False,
            eos_token=None,
        )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_file = output_dir / (input_dir.name + ".jsonl")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    for file in input_dir.iterdir():
        with jsonlines.open(file, "r") as reader:
            chunks = []
            for i, dataum in enumerate(reader):
                chunks += [dataum]

            if nsamples // 10 * seq > len(chunks):
                print(f"not enough data in trajectory: {file}, skipping", file=sys.stderr)
                continue

            # len(chunks) - (seq + 1) since we don't want to start too end into the trajectory
            # +1 in (seq + 1) and + 1 at the end to exclude the 0th index
            begginings = np.random.choice(len(chunks) - (seq + 1), size=nsamples) + 1

            for beggining in begginings:
                raw_histories = process_helper_raw_observation(chunks[beggining : beggining + seq])

                with jsonlines.open(output_file, "a") as writer:
                    writer.write_all([raw_histories])

    return 1


def worker(args):
    from_, to_, input_dirs, output_dir, seq, nsamples = args
    
    scores = []
    for i in range(from_, to_):
        input_dir = input_dirs[i]
        print(input_dir, file=sys.stderr)
        try:
            faster_load_and_process_chunks(
                input_dir=input_dir,
                output_dir=output_dir,
                seq=seq, 
                nsamples=nsamples,
            )
        except BaseException as e:
            print(''.join(traceback.format_exception(None, e, e.__traceback__)), file=sys.stderr)

    return scores


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument("--seq", default=128, type=int)
    parser.add_argument("--nsamples", default=500000, type=int)
    parser.add_argument("--num_workers", default=16, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    directories = list(Path(args.input_dir).iterdir())

    with Pool(args.num_workers) as pool:
        scores = list(
            pool.map(worker, [
                (
                    i * len(directories) // args.num_workers,
                    (i + 1) * len(directories) // args.num_workers,
                    directories,
                    args.output_dir,
                    args.seq,
                    args.nsamples,
                )
                for i in range(args.num_workers)
            ]
        ))
