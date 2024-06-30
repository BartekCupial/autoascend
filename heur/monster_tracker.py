import re

import numpy as np
from nle.nethack import actions as A

import heur.utils as utils
from heur.exceptions import AgentPanic
from heur.glyph import C, G

try:
    import numba as nb
except ImportError:

    class nb:
        b1 = bool
        njit = lambda *a, **k: (lambda f: f)


@nb.njit("b1[:,:](i2[:,:],i2[:,:],i4)", cache=True)
def disappearance_mask(old_mons, new_mons, max_radius):
    ret = np.zeros_like(new_mons, dtype=nb.b1)
    for y in range(new_mons.shape[0]):
        for x in range(new_mons.shape[1]):
            glyph = old_mons[y, x]
            if glyph == -1:
                continue
            ret[y, x] = (
                new_mons[
                    max(0, y - max_radius) : min(y + max_radius + 1, new_mons.shape[0]),
                    max(0, x - max_radius) : min(x + max_radius + 1, new_mons.shape[1]),
                ]
                != glyph
            ).all()
    return ret


@nb.njit("optional(b1[:,:])(i2[:,:],i2[:,:],i2[:,:],i4)", cache=True)
def figure_out_monster_movement(peaceful_mons, aggressive_mons, new_mons, max_radius):
    ret_peaceful_mons = np.zeros_like(peaceful_mons, dtype=nb.b1)
    for y in range(new_mons.shape[0]):
        for x in range(new_mons.shape[1]):
            glyph = new_mons[y, x]
            if glyph == -1:
                continue

            can_be_peaceful = False
            can_be_aggressive = False
            for py in range(max(0, y - max_radius), min(y + max_radius + 1, new_mons.shape[0])):
                for px in range(max(0, x - max_radius), min(x + max_radius + 1, new_mons.shape[1])):
                    if peaceful_mons[py, px] == glyph:
                        can_be_peaceful = True
                    if aggressive_mons[py, px] == glyph:
                        can_be_aggressive = True
            if can_be_peaceful == can_be_aggressive:
                return None
            if can_be_peaceful:
                ret_peaceful_mons[y, x] = True

    return ret_peaceful_mons


class MonsterTracker:
    def __init__(self, agent):
        self.agent = agent
        self.on_panic()

    def on_panic(self):
        self._last_glyphs = None
        self.peaceful_monster_mask = np.zeros((C.SIZE_Y, C.SIZE_X), bool)
        self.monster_mask = np.zeros((C.SIZE_Y, C.SIZE_X), bool)

    def take_all_monsters(self):
        if utils.any_in(self.agent.glyphs, G.SWALLOW):
            return {}
        with self.agent.atom_operation():
            self.agent.step(A.Command.WHATIS, iter(["M"]))
            if "No monsters are currently shown on the map." in self.agent.message:
                return {}
            try:
                index = self.agent.popup.index("All monsters currently shown on the map:")
            except IndexError:
                assert 0, (self.agent.message, self.agent.popup)
            regex = re.compile(r"^<(\d+),(\d+)>  ([\x00-\x7F])  ([a-zA-z-,' ]+)$")

            monsters = {}
            for line in self.agent.popup[index + 1 :]:
                r = regex.search(line)
                assert r is not None, line
                x, y, char, name = r.groups()
                y, x = int(y), int(x) - 1

                # char_on_map = self.agent.last_observation['chars'][y, x]
                # assert ord(char) == char_on_map, (char, chr(char_on_map))

                monsters[y, x] = name
        return monsters

    def _get_current_masks(self):
        new_monster_mask = utils.isin(self.agent.glyphs, G.MONS, G.INVISIBLE_MON)
        new_monster_mask[self.agent.blstats.y, self.agent.blstats.x] = 0
        pet_mask = utils.isin(self.agent.glyphs, G.PETS)

        return new_monster_mask, pet_mask

    def update(self):
        new_monster_mask, _ = self._get_current_masks()

        if self._last_glyphs is None:
            new_peaceful_mons = None
        else:
            pea_mon = self._last_glyphs.copy()
            pea_mon[~self.peaceful_monster_mask] = -1
            agr_mon = self._last_glyphs.copy()
            agr_mon[~self.monster_mask | self.peaceful_monster_mask] = -1
            new_mon = self.agent.glyphs.copy()
            new_mon[~new_monster_mask] = -1
            new_peaceful_mons = figure_out_monster_movement(pea_mon, agr_mon, new_mon, max_radius=2)

        self.monster_mask = new_monster_mask
        self.peaceful_monster_mask.fill(0)
        if not self.agent.character.prop.hallu:
            if new_peaceful_mons is None:
                all_monsters = self.take_all_monsters()
                (
                    self.monster_mask,
                    pet_mask,
                ) = self._get_current_masks()  # glyphs can change sometimes after calling `take_all_monsters`
                for (y, x), name in all_monsters.items():
                    if not (
                        self.monster_mask[y, x]
                        or pet_mask[y, x]
                        or (y, x) == (self.agent.blstats.y, self.agent.blstats.x)
                    ):
                        raise AgentPanic("monsters differs between list and glyphs")
                    if "peaceful" in name and not pet_mask[y, x]:
                        self.peaceful_monster_mask[y, x] = 1
            else:
                self.peaceful_monster_mask = new_peaceful_mons
        # TODO: on hallu no monsters are peaceful

        assert (~self.peaceful_monster_mask | self.monster_mask).all()
        self._last_glyphs = self.agent.glyphs.copy()
