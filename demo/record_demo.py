#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import ast
import contextlib
import os
import termios
import time
import timeit
import tty

import gym
import nle  # noqa: F401
from nle import nethack

from demo.wrappers import EnvWrapper, NLEDemo


@contextlib.contextmanager
def dummy_context():
    yield None


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


def get_action(env):
    if FLAGS.mode == "random":
        action = env.action_space.sample()
    elif FLAGS.mode == "human":
        while True:
            with no_echo():
                ch = ord(os.read(0, 1))
            if ch in [nethack.C("c")]:
                print("Received exit code {}. Aborting.".format(ch))
                return None
            try:
                action = env.actions.index(ch)
                break
            except ValueError:
                print(("Selected action '%s' is not in action list. Please try again.") % chr(ch))
                if not FLAGS.print_frames_separately:
                    print("\033[2A")  # Go up 2 lines.
                continue
    return action


def play():
    orig_env = gym.make(
        FLAGS.env,
        save_ttyrec_every=1,
        savedir=FLAGS.savedir,
    )

    env = NLEDemo(orig_env, FLAGS.demodir)

    if FLAGS.demopath:
        obs = env.load_from_file(FLAGS.demopath, FLAGS.demostep)
    else:
        obs = env.reset()

    steps = 0
    episodes = 0
    reward = 0.0
    action = None

    mean_sps = 0
    mean_reward = 0.0

    total_start_time = timeit.default_timer()
    start_time = total_start_time

    while True:
        if not FLAGS.no_render:
            env.render("human")

        break
        action = get_action(env)

        if action is None:
            break

        obs, reward, done, info = env.step(action)
        steps += 1

        mean_reward += (reward - mean_reward) / steps

        if not done:
            continue

        time_delta = timeit.default_timer() - start_time

        print("Final reward:", reward)
        print("End status:", info["end_status"].name)
        print("Mean reward:", mean_reward)

        sps = steps / time_delta
        print("Episode: %i. Steps: %i. SPS: %f" % (episodes, steps, sps))

        episodes += 1
        mean_sps += (sps - mean_sps) / episodes

        start_time = timeit.default_timer()

        steps = 0
        mean_reward = 0.0

        if episodes == FLAGS.ngames:
            break
        env.reset()
    env.close()
    print(
        "Finished after %i episodes and %f seconds. Mean sps: %f"
        % (episodes, timeit.default_timer() - total_start_time, mean_sps)
    )


def main():
    parser = argparse.ArgumentParser(description="NLE Play tool.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enables debug mode, which will drop stack into " "an ipdb shell if an exception is raised.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="human",
        choices=["human", "random"],
        help="Control mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="NetHackScore-v0",
        help="Gym environment spec. Defaults to 'NetHackStaircase-v0'.",
    )
    parser.add_argument(
        "-n",
        "--ngames",
        type=int,
        default=1,
        help="Number of games to be played before exiting. " "NetHack will auto-restart if > 1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Number of maximum steps per episode.",
    )
    parser.add_argument(
        "--savedir",
        default="nle_data/play_data",
        type=str,
        help="Directory path where data will be saved. " "Defaults to 'nle_data/play_data'.",
    )
    parser.add_argument(
        "--demodir",
        default="demo_data/play_data",
        type=str,
        help="Directory path where data will be saved. " "Defaults to 'demo_data/play_data'.",
    )
    parser.add_argument(
        "--demopath", default=None, type=str, help="If exists we will continue playing the demo from it."
    )
    parser.add_argument(
        "--demostep", default=-1, type=int, help="If demopath exists we will continue playing the demo from this step."
    )
    parser.add_argument("--no-render", action="store_true", help="Disables env.render().")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "full", "ansi"],
        help="Render mode. Defaults to 'human'.",
    )
    parser.add_argument(
        "--print-frames-separately",
        "-p",
        action="store_true",
        help="Don't overwrite frames, print them all.",
    )
    parser.add_argument(
        "--wizard",
        "-D",
        action="store_true",
        help="Use wizard mode.",
    )
    global FLAGS
    FLAGS = parser.parse_args()

    if FLAGS.debug:
        import ipdb

        cm = ipdb.launch_ipdb_on_exception
    else:
        cm = dummy_context

    with cm():
        if FLAGS.savedir == "args":
            FLAGS.savedir = "{}_{}_{}.zip".format(time.strftime("%Y%m%d-%H%M%S"), FLAGS.mode, FLAGS.env)
        elif FLAGS.savedir == "None":
            FLAGS.savedir = None  # Not saving any ttyrecs.

        play()


if __name__ == "__main__":
    main()
