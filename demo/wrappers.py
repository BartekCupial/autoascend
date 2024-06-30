import contextlib
import ctypes
import os
import pickle
import threading
import time
from pathlib import Path

import gym
import numpy as np
from gym import spaces
from nle import nethack as nh

from heur.agent import Agent
from heur.character import Character


class NLEDemo(gym.Wrapper):
    """
    Records actions taken, creates checkpoints, allows time travel, restoring and saving of states
    """

    def __init__(self, env, gamesavedir):
        super().__init__(env)
        self.action_space = spaces.Discrete(env.unwrapped.action_space.n + 1)  # add "time travel" action
        self.save_every_k = 100
        self.gamesavedir = gamesavedir
        self.savedir = Path(gamesavedir) / Path(self.env.nethack._ttyrec).stem

    def step(self, action):
        if self.steps_in_the_past > 0:
            self.restore_past_state()

        if len(self.done) > 0 and self.done[-1]:
            obs = self.obs[-1]
            reward = 0
            done = True
            info = None

        else:
            obs, reward, done, info = self.env.step(action)
            # self.env.render("human")

            self.actions.append(action)
            self.obs.append(obs)
            self.rewards.append(reward)
            self.done.append(done)
            self.info.append(info)

        # periodic checkpoint saving
        if not done:
            if (
                len(self.checkpoint_action_nr) > 0
                and len(self.actions) >= self.checkpoint_action_nr[-1] + self.save_every_k
            ) or (len(self.checkpoint_action_nr) == 0 and len(self.actions) >= self.save_every_k):
                self.save_checkpoint()

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # self.env.render("human")
        self.actions = []
        self.checkpoints = []
        self.checkpoint_action_nr = []
        self.obs = [obs]
        self.rewards = []
        self.done = [False]
        self.info = [None]
        self.steps_in_the_past = 0
        self.seeds = self.env.get_seeds()
        return obs

    def save_to_file(self):
        dat = {
            "actions": self.actions,
            "checkpoints": self.checkpoints,
            "checkpoint_action_nr": self.checkpoint_action_nr,
            "rewards": self.rewards,
            "seeds": self.seeds,
        }
        with open(self.savedir / "game.demo", "wb") as f:
            pickle.dump(dat, f)

    def load_from_file(self, file_name, demostep=-1):
        self.reset()
        with open(file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat["actions"]
        self.checkpoints = dat["checkpoints"]
        self.checkpoint_action_nr = dat["checkpoint_action_nr"]
        self.rewards = dat["rewards"]
        self.seeds = dat["seeds"]
        self.load_state_and_walk_forward(demostep=demostep)

    def save_checkpoint(self):
        i = len(self.actions)

        chk_pth = self.savedir / f"ckpt_{i}"
        self.env.save(gamesavedir=chk_pth)
        self.checkpoints.append(chk_pth)
        self.checkpoint_action_nr.append(len(self.actions))

    def restore_past_state(self):
        self.actions = self.actions[: -self.steps_in_the_past]
        while len(self.checkpoints) > 0 and self.checkpoint_action_nr[-1] > len(self.actions):
            self.checkpoints.pop()
            self.checkpoint_action_nr.pop()
        self.load_state_and_walk_forward()
        self.steps_in_the_past = 0

    def load_state_and_walk_forward(self, demostep=-1):
        self.env.seed(*self.seeds)
        if len(self.checkpoints) == 0:
            self.env.reset()
            time_step = 0
        else:
            if demostep != -1:
                idx = np.where(np.array(self.checkpoint_action_nr) <= demostep)[0][-1]
            else:
                idx = -1
            self.env.unwrapped.load(self.checkpoints[idx])

            # print(self.env.get_seeds())
            # import nle
            # obs = self.env.last_observation
            # tty_chars_idx = self.env._observation_keys.index("tty_chars")
            # tty_colors_idx = self.env._observation_keys.index("tty_colors")
            # tty_cursor_idx = self.env._observation_keys.index("tty_cursor")
            # print(
            #     nle.nethack.tty_render(
            #         obs[tty_chars_idx], obs[tty_colors_idx], obs[tty_cursor_idx]
            #     )
            # )

            time_step = self.checkpoint_action_nr[idx]

        for action in self.actions[time_step:demostep]:
            self.env.step(action)


class AgentStepTimeout(KeyboardInterrupt):
    # it inheirits from KeyboardInterrupt because agent never catches it
    pass


class EnvWrapper(gym.Wrapper):
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
