import gym

from demo.utils.blstats import BLStats


class FinalStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, done_only=False):
        super().__init__(env)
        self.done_only = done_only

    def reset(self, **kwargs):
        self.step_num = 0
        self.max_dlvl = 1
        self.visited_sokoban = False
        self.sokoban_entry = None
        self.previous_dlvl = 1
        return self.env.reset(**kwargs)

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, done, info = self.env.step(action)
        self.step_num += 1

        if done or not self.done_only:
            info["episode_extra_stats"] = self.episode_extra_stats(info, last_observation)

        return obs, reward, done, info

    def episode_extra_stats(self, info, observation):
        extra_stats = info.get("episode_extra_stats", {})

        blstats = BLStats(*observation[self.env.unwrapped._blstats_index])

        # in sokoban we have to calculate max dlvl differently
        if blstats.dungeon_number == 4:
            # if first time in Sokoban
            if not self.visited_sokoban:
                self.sokoban_entry = self.previous_dlvl
                self.visited_sokoban = True

            # if first soko level 5 - 4 = +1
            # if last soko level 5 - 1 = +4
            self.sokoban_level_number = self.sokoban_entry + (5 - blstats.level_number)

            self.max_dlvl = max(self.max_dlvl, self.sokoban_level_number)
        else:
            self.max_dlvl = max(self.max_dlvl, blstats.level_number)

        self.previous_dlvl = blstats.level_number

        blstats = blstats._asdict()
        include = [
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
            "hitpoints",
            "max_hitpoints",
            "gold",
            "energy",
            "max_energy",
            "armor_class",
            "experience_level",
            "experience_points",
            "score",
        ]
        blstats_info = dict(filter(lambda item: item[0] in include, blstats.items()))

        return {**extra_stats, **blstats_info, "step": self.step_num, "max_dlvl": self.max_dlvl}
