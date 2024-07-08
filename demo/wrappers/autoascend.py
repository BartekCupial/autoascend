import contextlib
import ctypes
import threading
import time

import gym
from nle import nethack as nh

from heur.agent import Agent


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
