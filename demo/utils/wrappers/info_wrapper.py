import gym


class InfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, done_only=True):
        super().__init__(env)
        self.done_only = done_only

    def add_more_stats(self, info, last_observation, done):
        if done or not self.done_only:
            info["episode_extra_stats"] = self.episode_extra_stats(info, last_observation)

    def episode_extra_stats(self, info, last_observation):
        raise NotImplementedError
