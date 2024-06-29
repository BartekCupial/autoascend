import contextlib
import functools
import re
from functools import partial
from itertools import chain

import nle.nethack as nh
import numpy as np
from nle.nethack import actions as A

import heur.objects as O
import heur.utils as utils
from heur.character import Character
from heur.exceptions import AgentPanic
from heur.glyph import WEA, G, MON
from heur.strategy import Strategy


def flatten_items(iterable):
    ret = []
    for item in iterable:
        ret.append(item)
        if item.is_container():
            ret.extend(flatten_items(item.content))
    return ret


def find_equivalent_item(item, iterable):
    assert item.text
    for i in iterable:
        assert i.text
        if i.text == item.text:
            return i
    assert 0, (item, iterable)


def check_if_triggered_container_trap(message):
    return ('A cloud of ' in message and ' gas billows from ' in message) or \
            'Suddenly you are frozen in place!' in message or \
            'A tower of flame bursts from ' in message or \
            'You are jolted by a surge of electricity!' in message or \
            'But luckily ' in message


class Item:
    # beatitude
    UNKNOWN = 0
    CURSED = 1
    UNCURSED = 2
    BLESSED = 3

    # shop status
    NOT_SHOP = 0
    FOR_SALE = 1
    UNPAID = 2

    def __init__(self, objs, glyphs, count=1, status=UNKNOWN, modifier=None, equipped=False, at_ready=False,
                 monster_id=None, shop_status=NOT_SHOP, price=0, dmg_bonus=None, to_hit_bonus=None,
                 naming='', comment='', uses=None, text=None):
        assert isinstance(objs, list) and len(objs) >= 1
        assert isinstance(glyphs, list) and len(glyphs) >= 1 and all((nh.glyph_is_object(g) for g in glyphs))
        assert isinstance(count, int)

        self.objs = objs
        self.glyphs = glyphs
        self.count = count
        self.status = status
        self.modifier = modifier
        self.equipped = equipped
        self.uses = uses
        self.at_ready = at_ready
        self.monster_id = monster_id
        self.shop_status = shop_status
        self.price = price
        self.dmg_bonus = dmg_bonus
        self.to_hit_bonus = to_hit_bonus
        self.naming = naming
        self.comment = comment
        self.text = text

        self.content = None  # for checked containers it will be set after the constructor
        self.container_id = None  # for containers and possible containers it will be set after the constructor

        self.category = O.get_category(self.objs[0])
        assert all((ord(nh.objclass(nh.glyph_to_obj(g)).oc_class) == self.category for g in self.glyphs))

    def display_glyphs(self):
        if self.is_corpse():
            assert self.monster_id is not None
            return [nh.GLYPH_BODY_OFF + self.monster_id]
        if self.is_statue():
            assert self.monster_id is not None
            return [nh.GLYPH_STATUE_OFF + self.monster_id]
        return self.glyphs

    def is_unambiguous(self):
        return len(self.objs) == 1

    def can_be_dropped_from_inventory(self):
        return not (
            (isinstance(self.objs[0], (O.Weapon, O.WepTool)) and self.status == Item.CURSED and self.equipped) or
            (isinstance(self.objs[0], O.Armor) and self.equipped) or
            (self.is_unambiguous() and self.object == O.from_name('loadstone') and self.status == Item.CURSED) or
            (self.category == nh.BALL_CLASS and self.equipped)
        )

    def weight(self, with_content=True):
        return self.count * self.unit_weight(with_content=with_content)

    def unit_weight(self, with_content=True):
        if self.is_corpse():
            return MON.permonst(self.monster_id).cwt

        if self.is_possible_container():
            return 100000

        if self.objs[0] in [
            O.from_name("glob of gray ooze"),
            O.from_name("glob of brown pudding"),
            O.from_name("glob of green slime"),
            O.from_name("glob of black pudding"),
        ]:
            assert self.is_unambiguous()
            return 10000  # weight is unknown

        weight = max((obj.wt for obj in self.objs))

        if self.is_container() and with_content:
            weight += self.content.weight()  # TODO: bag of holding

        return weight

    @property
    def object(self):
        assert self.is_unambiguous()
        return self.objs[0]

    ######## WEAPON

    def is_weapon(self):
        return self.category == nh.WEAPON_CLASS

    def get_weapon_bonus(self, large_monster):
        assert self.is_weapon()

        hits = []
        dmgs = []
        for weapon in self.objs:
            dmg = WEA.expected_damage(weapon.damage_large if large_monster else weapon.damage_small)
            to_hit = 1 + weapon.hitbon
            if self.modifier is not None:
                dmg += max(0, self.modifier)
                to_hit += self.modifier

            dmg += 0 if self.dmg_bonus is None else self.dmg_bonus
            to_hit += 0 if self.to_hit_bonus is None else self.to_hit_bonus

            dmgs.append(dmg)
            hits.append(to_hit)

        # assume the worse
        return min(hits), min(dmgs)

    def is_launcher(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        return self.object.name in ['bow', 'elven bow', 'orcish bow', 'yumi', 'crossbow', 'sling']

    def is_fired_projectile(self, launcher=None):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        arrows = ['arrow', 'elven arrow', 'orcish arrow', 'silver arrow', 'ya']

        if launcher is None:
            return self.object.name in (arrows + ['crossbow bolt'])  # TODO: sling ammo
        else:
            launcher_name = launcher.object.name
            if launcher_name == 'crossbow':
                return self.object.name == 'crossbow bolt'
            elif launcher_name == 'sling':
                # TODO: sling ammo
                return False
            else:  # any bow
                assert launcher_name in ['bow', 'elven bow', 'orcish bow', 'yumi'], launcher_name
                return self.object.name in arrows

    def is_thrown_projectile(self):
        if not self.is_weapon() or not self.is_unambiguous():
            return False

        # TODO: boomerang
        # TODO: aklys, Mjollnir
        return self.object.name in \
               ['dagger', 'orcish dagger', 'dagger silver', 'athame dagger', 'elven dagger',
                'worm tooth', 'knife', 'stiletto', 'scalpel', 'crysknife',
                'dart', 'shuriken']

    def __str__(self):
        if self.text is not None:
            return self.text
        return (f'{self.count}_'
                f'{self.status if self.status is not None else ""}_'
                f'{self.modifier if self.modifier is not None else ""}_'
                f'{",".join(list(map(lambda x: x.name, self.objs)))}'
                )

    def __repr__(self):
        return str(self)

    ######## ARMOR

    def is_armor(self):
        return self.category == nh.ARMOR_CLASS

    def get_ac(self):
        assert self.is_armor()
        return self.object.ac - (self.modifier if self.modifier is not None else 0)


    ######## WAND

    def is_wand(self):
        return isinstance(self.objs[0], O.Wand)

    def is_beam_wand(self):
        if not self.is_wand():
            return False
        beam_wand_types = ['cancellation', 'locking', 'make invisible',
                           'nothing', 'opening', 'polymorph', 'probing', 'slow monster',
                           'speed monster', 'striking', 'teleportation', 'undead turning']
        beam_wand_types = [O.from_name(w, nh.WAND_CLASS) for w in beam_wand_types]
        for obj in self.objs:
            if obj not in beam_wand_types:
                return False
        return True

    def is_ray_wand(self):
        if not self.is_wand():
            return False
        ray_wand_types = ['cold', 'death', 'digging', 'fire', 'lightning', 'magic missile', 'sleep']
        ray_wand_types = [O.from_name(w, nh.WAND_CLASS) for w in ray_wand_types]
        for obj in self.objs:
            if obj not in ray_wand_types:
                return False
        return True

    def wand_charges_left(self, item):
        assert item.is_wand()

    def is_offensive_usable_wand(self):
        if len(self.objs) != 1:
            return False
        if not self.is_ray_wand():
            return False
        if self.uses == 'no charges':
            # TODO: is it right ?
            return False
        if self.objs[0] == O.from_name('sleep', nh.WAND_CLASS):
            return False
        if self.objs[0] == O.from_name('digging', nh.WAND_CLASS):
            return False
        return True

    ######## FOOD

    def is_food(self):
        if isinstance(self.objs[0], O.Food):
            assert self.is_unambiguous()
            return True

    def nutrition_per_weight(self):
        # TODO: corpses/tins
        assert self.is_food()
        return self.object.nutrition / max(self.unit_weight(), 1)

    def is_corpse(self):
        if self.objs[0] == O.from_name('corpse'):
            assert self.is_unambiguous()
            return True
        return False

    ######## STATUE

    def is_statue(self):
        if self.objs[0] == O.from_name('statue'):
            assert self.is_unambiguous()
            return True
        return False

    ######## CONTAINER

    def is_chest(self):
        if self.is_unambiguous() and self.object.name == 'bag of tricks':
            return False
        assert self.is_possible_container() or self.is_container(), self.objs
        assert isinstance(self.objs[0], O.Container), self.objs
        return self.objs[0].desc != 'bag'

    def is_container(self):
        # don't consider bag of tricks as a container.
        # If the identifier doesn't exist yet, it's not consider a container
        return self.content is not None

    def is_possible_container(self):
        if self.is_container():
            return False

        if self.is_unambiguous() and self.object.name == 'bag of tricks':
            return False
        return any((isinstance(obj, O.Container) for obj in self.objs))

    def content(self):
        assert self.is_container()
        return self.content


class ContainerContent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.items = []
        self.locked = False

    def __iter__(self):
        return iter(self.items)

    def weight(self):
        if self.locked:
            return 100000
        return sum((item.weight() for item in self.items))


class ItemManager:
    def __init__(self, agent):
        self.agent = agent
        self.object_to_glyph = {}
        self.glyph_to_object = {}
        self._last_object_glyph_mapping_update_step = None
        self._glyph_to_price_range = {}

        self._is_not_bag_of_tricks = set()

        # the container content should be edited instead of creating new one if exists.
        # Item.content keeps reference to it
        self.container_contents = {}  # container_id -> ContainerContent
        self._last_container_identifier = 0

        self._glyph_to_possible_wand_types = {}
        self._already_engraved_glyphs = set()

    def on_panic(self):
        self.update_object_glyph_mapping()

    def update(self):
        if self._last_object_glyph_mapping_update_step is None or \
                self._last_object_glyph_mapping_update_step + 200 < self.agent.step_count:
            self.update_object_glyph_mapping()

    def update_object_glyph_mapping(self):
        with self.agent.atom_operation():
            self.agent.step(A.Command.KNOWN)
            for line in self.agent.popup:
                if line.startswith('*'):
                    assert line[1] == ' ' and line[-1] == ')' and line.count('(') == 1 and line.count(')') == 1, line
                    name = line[1: line.find('(')].strip()
                    desc = line[line.find('(') + 1: -1].strip()

                    n_objs, n_glyphs = ItemManager.parse_name(name)
                    d_glyphs = O.desc_to_glyphs(desc, O.get_category(n_objs[0]))
                    assert d_glyphs
                    if len(n_objs) == 1 and len(d_glyphs) == 1:
                        obj, glyph = n_objs[0], d_glyphs[0]

                        assert glyph not in self.glyph_to_object or self.glyph_to_object[glyph] == obj
                        self.glyph_to_object[glyph] = obj
                        assert obj not in self.object_to_glyph or self.object_to_glyph[obj] == glyph
                        self.object_to_glyph[obj] = glyph

            self._last_object_glyph_mapping_update_step = self.agent.step_count

    def _get_new_container_identifier(self):
        ret = self._last_container_identifier
        self._last_container_identifier += 1
        return str(ret)

    def _buy_price_identification(self):
        if self.agent.blstats.charisma <= 5:
            charisma_multiplier = 2
        elif self.agent.blstats.charisma <= 7:
            charisma_multiplier = 1.5
        elif self.agent.blstats.charisma <= 10:
            charisma_multiplier = 1 + 1 / 3
        elif self.agent.blstats.charisma <= 15:
            charisma_multiplier = 1
        elif self.agent.blstats.charisma <= 17:
            charisma_multiplier = 3 / 4
        elif self.agent.blstats.charisma <= 18:
            charisma_multiplier = 2 / 3
        else:
            charisma_multiplier = 1 / 2

        if (self.agent.character.role == Character.TOURIST and self.agent.blstats.experience_level < 15) or \
                (self.agent.inventory.items.shirt is not None and self.agent.inventory.items.suit is None and
                 self.agent.inventory.items.cloak is None) or \
                (self.agent.inventory.items.helm is not None and self.agent.inventory.items.helm.is_unambiguous() and
                 self.agent.inventory.items.helm.object == O.from_name('dunce cap')):
            dupe_multiplier = (4 / 3, 4 / 3)
        elif self.agent.inventory.items.helm is not None and \
                any(((obj == O.from_name('dunce cap')) for obj in self.agent.inventory.items.helm.objs)):
            dupe_multiplier = (4 / 3, 1)
        else:
            dupe_multiplier = (1, 1)

        for item in self.agent.inventory.items_below_me:
            if item.shop_status == Item.FOR_SALE and not item.is_unambiguous() and len(item.glyphs) == 1 and \
                    isinstance(item.objs[0], (O.Armor, O.Ring, O.Wand, O.Scroll, O.Potion, O.Tool)):
                # TODO: not an artifact
                # TODO: Base price of partly eaten food, uncursed water, and (x:-1) wands is 0.
                assert item.price % item.count == 0
                low = int(item.price / item.count / charisma_multiplier / (4 / 3) / dupe_multiplier[0])
                if isinstance(item.objs[0], (O.Weapon, O.Armor)):
                    low = 0  # +10 zorkmoids for each point of enchantment

                if low <= 5:
                    low = 0
                high = int(item.price / item.count / charisma_multiplier / dupe_multiplier[1]) + 1
                if item.glyphs[0] in self._glyph_to_price_range:
                    l, h = self._glyph_to_price_range[item.glyphs[0]]
                    low = max(low, l)
                    high = min(high, h)
                assert low <= high, (low, high)
                self._glyph_to_price_range[item.glyphs[0]] = (low, high)

                # update mapping for that object
                self.possible_objects_from_glyph(item.glyphs[0])

    def price_identification(self):
        if self.agent.character.prop.hallu:
            return
        if not self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            return

        self._buy_price_identification()

    def update_possible_objects(self, item):
        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in item.glyphs))
        item.objs = [o for o in item.objs if o in possibilities_from_glyphs]
        assert len(item.objs)

    def get_item_from_text(self, text, category=None, glyph=None, *, position):
        # position acts as a container identifier if the container is not called. If the item is in inventory set it to None

        if self.agent.character.prop.hallu:
            glyph = None

        try:
            objs, glyphs, count, status, modifier, *args = \
                self.parse_text(text, category, glyph)
            category = O.get_category(objs[0])
        except:
            # TODO: when blind, it may not work as expected, e.g. "a shield", "a gem", "a potion", etc
            if self.agent.character.prop.blind:
                obj = O.from_name('unknown')
                glyphs = O.possible_glyphs_from_object(obj)
                return Item([obj], glyphs, text=text)
            raise


        possibilities_from_glyphs = set.union(*(set(self.possible_objects_from_glyph(glyph)) for glyph in glyphs))
        objs = [o for o in objs if o in possibilities_from_glyphs]
        assert len(objs), ([O.objects[g - nh.GLYPH_OBJ_OFF].desc for g in glyphs],
                           [o.name for o in possibilities_from_glyphs], text)

        if status == Item.UNKNOWN and (
                self.agent.character.role == Character.PRIEST or
                (modifier is not None and category not in [nh.ARMOR_CLASS, nh.RING_CLASS])):
            # TODO: amulets of yendor
            status = Item.UNCURSED

        old_objs = None
        old_glyphs = None
        while old_objs != objs or old_glyphs != glyphs:
            old_objs = objs
            old_glyphs = glyphs

            objs = [o for o in objs if o not in self.object_to_glyph or self.object_to_glyph[o] in glyphs]
            glyphs = [g for g in glyphs if g not in self.glyph_to_object or self.glyph_to_object[g] in objs]
            if len(objs) == 1 and len(glyphs) == 1:
                assert glyphs[0] not in self.glyph_to_object or self.glyph_to_object[glyphs[0]] == objs[0]
                self.glyph_to_object[glyphs[0]] = objs[0]
                assert objs[0] not in self.object_to_glyph or self.object_to_glyph[objs[0]] == glyphs[0]
                self.object_to_glyph[objs[0]] = glyphs[0]
            elif len(objs) == 1 and objs[0] in self.object_to_glyph:
                glyphs = [self.object_to_glyph[objs[0]]]
            elif len(glyphs) == 1 and glyphs[0] in self.glyph_to_object:
                objs = [self.glyph_to_object[glyphs[0]]]

        item = Item(objs, glyphs, count, status, modifier, *args, text)

        if item.is_possible_container() or item.is_container():
            if item.comment:
                identifier = item.comment
            else:
                if position is not None:
                    identifier = position
                else:
                    identifier = self._get_new_container_identifier()
            item.container_id = identifier
            if identifier in self.container_contents:
                item.content = self.container_contents[identifier]
                if len(item.glyphs) == 1 and item.glyphs[0] not in self._is_not_bag_of_tricks:
                    self._is_not_bag_of_tricks.add(item.glyphs[0])
                    self.update_possible_objects(item)


        # FIXME: it gives a better score. Implement it in item equipping
        if item.status == Item.UNKNOWN:
            item.status = Item.UNCURSED

        return item

    def possible_objects_from_glyph(self, glyph):
        """ Get possible objects and update glyph_to_object and object_to_glyph when object isn't unambiguous.
        """
        assert nh.glyph_is_object(glyph)
        if glyph in self.glyph_to_object:
            return [self.glyph_to_object[glyph]]

        objs = []
        for obj in O.possibilities_from_glyph(glyph):
            if obj in self.object_to_glyph:
                continue
            if glyph in self._glyph_to_price_range and hasattr(obj, 'cost') and \
                    not self._glyph_to_price_range[glyph][0] <= obj.cost <= self._glyph_to_price_range[glyph][1]:
                continue
            if glyph in self._glyph_to_possible_wand_types and obj not in self._glyph_to_possible_wand_types[glyph]:
                continue
            if glyph in self._is_not_bag_of_tricks and obj.name == 'bag of tricks':
                continue
            objs.append(obj)

        if len(objs) == 1:
            self.glyph_to_object[glyph] = objs[0]
            assert objs[0] not in self.object_to_glyph
            self.object_to_glyph[objs[0]] = glyph

            # update objects with have the same possible glyph
            for g in O.possible_glyphs_from_object(objs[0]):
                self.possible_objects_from_glyph(g)
        assert len(objs), (O.objects[glyph - nh.GLYPH_OBJ_OFF].desc, self._glyph_to_price_range[glyph])
        return objs

    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 1024)
    def parse_text(text, category=None, glyph=None):
        assert glyph is None or nh.glyph_is_normal_object(glyph), glyph

        if category is None and glyph is not None:
            category = ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)
        assert glyph is None or category is None or category == ord(nh.objclass(nh.glyph_to_obj(glyph)).oc_class)

        assert category not in [nh.RANDOM_CLASS]

        matches = re.findall(
            r'^(a|an|the|\d+)'
            r'( empty)?'
            r'( (cursed|uncursed|blessed))?'
            r'( (very |thoroughly )?(rustproof|poisoned|corroded|rusty|burnt|rotted|partly eaten|partly used|diluted|unlocked|locked|wet|greased))*'
            r'( ([+-]\d+))? '
            r"([a-zA-z0-9-!'# ]+)"
            r'( \(([0-9]+:[0-9]+|no charge)\))?'
            r'( \(([a-zA-Z0-9; ]+(, flickering|, gleaming|, glimmering)?[a-zA-Z0-9; ]*)\))?'
            r'( \((for sale|unpaid), (\d+ aum, )?((\d+)[a-zA-Z- ]+|no charge)\))?'
            r'$',
            text)
        assert len(matches) <= 1, text
        assert len(matches), (text, len(text))

        (
            count,
            effects1,
            _, status,
            effects2, _, _,
            _, modifier,
            name,
            _, uses,
            _, info, _,
            _, shop_status, _, _, shop_price
        ) = matches[0]
        # TODO: effects, uses

        if info in {'being worn', 'being worn; slippery', 'wielded', 'chained to you'} or info.startswith('weapon in ') or \
                info.startswith('tethered weapon in '):
            equipped = True
            at_ready = False
        elif info in {'at the ready', 'in quiver', 'in quiver pouch', 'lit'}:
            equipped = False
            at_ready = True
        elif info in {'', 'alternate weapon; not wielded', 'alternate weapon; notwielded'}:
            equipped = False
            at_ready = False
        else:
            assert 0, info

        if not shop_price:
            shop_price = 0
        else:
            shop_price = int(shop_price)

        count = int({'a': 1, 'an': 1, 'the': 1}.get(count, count))
        status = {'': Item.UNKNOWN, 'cursed': Item.CURSED, 'uncursed': Item.UNCURSED, 'blessed': Item.BLESSED}[status]
        # TODO: should be uses -- but the score is better this way
        if uses and status == Item.UNKNOWN:
            status = Item.UNCURSED
        modifier = None if not modifier else {'+': 1, '-': -1}[modifier[0]] * int(modifier[1:])
        monster_id = None

        if ' containing ' in name:
            # TODO: use number of items for verification
            name = name[:name.index(' containing ')]

        comment = ''
        naming = ''
        if ' named ' in name:
            # TODO: many of these are artifacts
            naming = name[name.index(' named ') + len(' named '):]
            name = name[:name.index(' named ')]
            if '#' in naming:
                # all given names by the bot, starts with #
                pos = naming.index('#')
                comment = naming[pos + 1:]
                naming = naming[:pos]
            else:
                comment = ''

        if shop_status == '':
            shop_status = Item.NOT_SHOP
        elif shop_status == 'for sale':
            shop_status = Item.FOR_SALE
        elif shop_status == 'unpaid':
            shop_status = Item.UNPAID
        else:
            assert 0, shop_status

        if name in ['potion of holy water', 'potions of holy water']:
            name = 'potion of water'
            status = Item.BLESSED
        elif name in ['potion of unholy water', 'potions of unholy water']:
            name = 'potion of water'
            status = Item.CURSED
        elif name in ['gold piece', 'gold pieces']:
            status = Item.UNCURSED

        # TODO: pass to Item class instance
        if name.startswith('tin of ') or name.startswith('tins of '):
            mon_name = name[len('tin of '):].strip()
            if mon_name.endswith(' meat'):
                mon_name = mon_name[:-len(' meat')]
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            if mon_name == 'spinach':
                monster_id = None
            else:
                monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'tin'
        elif name.endswith(' corpse') or name.endswith(' corpses'):
            mon_name = name[:name.index('corpse')].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'corpse'
        elif name.startswith('statue of ') or name.startswith('statues of ') or \
                name.startswith('historic statue of ') or name.startswith('historic statues of '):
            if name.startswith('historic'):
                mon_name = name[len('historic statue of '):].strip()
            else:
                mon_name = name[len('statue of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'statue'
        elif name.startswith('figurine of ') or name.startswith('figurines of '):
            mon_name = name[len('figurine of '):].strip()
            if mon_name.startswith('a '):
                mon_name = mon_name[2:]
            if mon_name.startswith('an '):
                mon_name = mon_name[3:]
            monster_id = nh.glyph_to_mon(MON.from_name(mon_name))
            name = 'figurine'
        elif name in ['novel', 'paperback', 'paperback book']:
            name = 'spellbook of novel'
        elif name.endswith(' egg') or name.endswith(' eggs'):
            monster_id = nh.glyph_to_mon(MON.from_name(name[:-len(' egg')].strip()))
            name = 'egg'
        elif name == 'worm teeth':
            name = 'worm tooth'

        dmg_bonus, to_hit_bonus = None, None

        if naming:
            if naming in ['Hachi', 'Idefix', 'Slasher', 'Sirius']:  # pet names
                pass
            elif name == 'corpse':
                pass
            elif name == 'spellbook of novel':
                pass
            else:
                name = naming

        if name == 'Excalibur':
            name = 'long sword'
            dmg_bonus = 5.5  # 1d10
            to_hit_bonus = 3  # 1d5
        elif name == 'Mjollnir':
            name = 'war hammer'
            dmg_bonus = 12.5  # 1d24
            to_hit_bonus = 3  # 1d5
        elif name == 'Cleaver':
            name = 'battle-axe'
            dmg_bonus = 3.5  # 1d6
            to_hit_bonus = 2  # 1d3
        elif name == 'Sting':
            name = 'elven dagger'
            dmg_bonus = 2.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Grimtooth':
            name = 'orcish dagger'
            dmg_bonus = 3.5  # +1d6
            to_hit_bonus = 1.5  # 1d2
        elif name in ['Sunsword', 'Frost Brand', 'Fire Brand', 'Demonbane', 'Giantslayer']:
            name = 'long sword'
            dmg_bonus = 5.5  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Vorpal Blade':
            name = 'long sword'
            dmg_bonus = 1
            to_hit_bonus = 3  # 1d5
        elif name == 'Orcrist':
            name = 'elven broadsword'
            dmg_bonus = 6  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Magicbane':
            name = 'athame'
            dmg_bonus = 7.25  # TODO
            to_hit_bonus = 2  # 1d3
        elif name in ['Grayswandir', 'Werebane']:
            name = 'silver saber'
            dmg_bonus = 15  # TODO: x2 + 1d20
            to_hit_bonus = 3  # 1d5
        elif name == 'Stormbringer':
            name = 'runesword'
            dmg_bonus = 6  # 1d2+1d8
            to_hit_bonus = 3  # 1d5
        elif name == 'Snickersnee':
            name = 'katana'
            dmg_bonus = 4.5  # 1d8
            to_hit_bonus = 1  # +1
        elif name == 'Trollsbane':
            name = 'morning star'
            dmg_bonus = 5.25  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Ogresmasher':
            name = 'war hammer'
            dmg_bonus = 3  # TODO: x2
            to_hit_bonus = 3  # 1d5
        elif name == 'Dragonbane':
            name = 'broadsword'
            dmg_bonus = 4.75  # TODO: x2
            to_hit_bonus = 3  # 1d5

        objs, ret_glyphs = ItemManager.parse_name(name)
        assert category is None or category == O.get_category(objs[0]), (text, category, O.get_category(objs[0]))

        if glyph is not None:
            assert glyph in ret_glyphs
            pos = O.possibilities_from_glyph(glyph)
            if objs[0].name not in ['elven broadsword', 'runed broadsword']:
                assert all(map(lambda o: o in pos, objs)), (objs, pos)
            ret_glyphs = [glyph]
            objs = sorted(set(objs).intersection(O.possibilities_from_glyph(glyph)))

        return (
            objs, ret_glyphs, count, status, modifier, equipped, at_ready, monster_id, shop_status, shop_price,
            dmg_bonus, to_hit_bonus, naming, comment, uses
        )

    @staticmethod
    @utils.copy_result
    @functools.lru_cache(1024 * 256)
    def parse_name(name):
        if name == 'wakizashi':
            name = 'short sword'
        elif name == 'ninja-to':
            name = 'broadsword'
        elif name == 'nunchaku':
            name = 'flail'
        elif name == 'shito':
            name = 'knife'
        elif name == 'naginata':
            name = 'glaive'
        elif name == 'gunyoki':
            name = 'food ration'
        elif name == 'osaku':
            name = 'lock pick'
        elif name == 'tanko':
            name = 'plate mail'
        elif name in ['pair of yugake', 'yugake']:
            name = 'pair of leather gloves'
        elif name == 'kabuto':
            name = 'helmet'
        elif name in ['flint stone', 'flint stones']:
            name = 'flint'
        elif name in ['unlabeled scroll', 'unlabeled scrolls', 'blank paper']:
            name = 'scroll of blank paper'
        elif name == 'eucalyptus leaves':
            name = 'eucalyptus leaf'
        elif name == 'pair of lenses':
            name = 'lenses'
        elif name.startswith('small glob'):
            name = name[len('small '):]
        elif name == 'knives':
            name = 'knife'

        # object identified (look on names)
        obj_ids = set()
        prefixes = [
            ('scroll of ', nh.SCROLL_CLASS),
            ('scrolls of ', nh.SCROLL_CLASS),
            ('spellbook of ', nh.SPBOOK_CLASS),
            ('spellbooks of ', nh.SPBOOK_CLASS),
            ('ring of ', nh.RING_CLASS),
            ('rings of ', nh.RING_CLASS),
            ('wand of ', nh.WAND_CLASS),
            ('wands of ', nh.WAND_CLASS),
            ('', nh.AMULET_CLASS),
            ('potion of ', nh.POTION_CLASS),
            ('potions of ', nh.POTION_CLASS),
            ('', nh.GEM_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
            ('', nh.COIN_CLASS),
            ('', nh.ROCK_CLASS),
        ]
        suffixes = [
            ('s', nh.GEM_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
            ('s', nh.COIN_CLASS),
        ]
        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and name == pref + obj_name:
                        obj_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_name = nh.objdescr.from_idx(i).oc_name
                    if obj_name and (name == obj_name + suf or \
                                     (c == nh.FOOD_CLASS and \
                                      name == obj_name.split()[0] + suf + ' ' + ' '.join(obj_name.split()[1:]))):
                        obj_ids.add(i)

        # object unidentified (look on descriptions)
        appearance_ids = set()
        prefixes = [
            ('scroll labeled ', nh.SCROLL_CLASS),
            ('scrolls labeled ', nh.SCROLL_CLASS),
            ('', nh.ARMOR_CLASS),
            ('pair of ', nh.ARMOR_CLASS),
            ('', nh.WEAPON_CLASS),
            ('', nh.TOOL_CLASS),
            ('', nh.FOOD_CLASS),
            ('', nh.BALL_CLASS),
        ]
        suffixes = [
            (' amulet', nh.AMULET_CLASS),
            (' amulets', nh.AMULET_CLASS),
            (' gem', nh.GEM_CLASS),
            (' gems', nh.GEM_CLASS),
            (' stone', nh.GEM_CLASS),
            (' stones', nh.GEM_CLASS),
            (' potion', nh.POTION_CLASS),
            (' potions', nh.POTION_CLASS),
            (' spellbook', nh.SPBOOK_CLASS),
            (' spellbooks', nh.SPBOOK_CLASS),
            (' ring', nh.RING_CLASS),
            (' rings', nh.RING_CLASS),
            (' wand', nh.WAND_CLASS),
            (' wands', nh.WAND_CLASS),
            ('s', nh.ARMOR_CLASS),
            ('s', nh.WEAPON_CLASS),
            ('s', nh.TOOL_CLASS),
            ('s', nh.FOOD_CLASS),
        ]

        for i in range(nh.NUM_OBJECTS):
            for pref, c in prefixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == pref + obj_descr:
                        appearance_ids.add(i)

            for suf, c in suffixes:
                if ord(nh.objclass(i).oc_class) == c:
                    obj_descr = nh.objdescr.from_idx(i).oc_descr
                    if obj_descr and name == obj_descr + suf:
                        appearance_ids.add(i)

        appearance_ids = list(appearance_ids)
        assert len(appearance_ids) == 0 or len({ord(nh.objclass(i).oc_class) for i in appearance_ids}), name

        #assert (len(obj_ids) > 0) ^ (len(appearance_ids) > 0), (name, obj_ids, appearance_ids)
        if (len(obj_ids) > 0) == (len(appearance_ids) > 0):
            return [O.from_name('unknown')], O.possible_glyphs_from_object(O.from_name('unknown'))

        if obj_ids:
            assert len(obj_ids) == 1, name
            obj_id = list(obj_ids)[0]
            objs = [O.objects[obj_id]]
            glyphs = [i for i in range(nh.GLYPH_OBJ_OFF, nh.NUM_OBJECTS + nh.GLYPH_OBJ_OFF)
                      if O.objects[i - nh.GLYPH_OBJ_OFF] is not None and objs[0] in O.possibilities_from_glyph(i)]
        else:
            glyphs = [obj_id + nh.GLYPH_OBJ_OFF for obj_id in appearance_ids]
            obj_id = list(appearance_ids)[0]
            glyph = obj_id + nh.GLYPH_OBJ_OFF
            objs = sorted(set.union(*[set(O.possibilities_from_glyph(i)) for i in glyphs]))
            assert name == 'runed broadsword' or \
                   all(map(lambda i: sorted(O.possibilities_from_glyph(i + nh.GLYPH_OBJ_OFF)) == objs, appearance_ids)), \
                name

        return objs, glyphs


class InventoryItems:
    def __init__(self, agent):
        self.agent = agent
        self._previous_inv_strs = None

        self._clear()

    def _clear(self):
        self.main_hand = None
        self.off_hand = None
        self.suit = None
        self.helm = None
        self.gloves = None
        self.boots = None
        self.cloak = None
        self.shirt = None

        self.total_weight = 0

        self.all_items = []
        self.all_letters = []

        self._recheck_containers = True

    def __iter__(self):
        return iter(self.all_items)

    def __str__(self):
        return (
                f'main_hand: {self.main_hand}\n'
                f'off_hand : {self.off_hand}\n'
                f'suit     : {self.suit}\n'
                f'helm     : {self.helm}\n'
                f'gloves   : {self.gloves}\n'
                f'boots    : {self.boots}\n'
                f'cloak    : {self.cloak}\n'
                f'shirt    : {self.shirt}\n'
                f'Items:\n' +
                '\n'.join([f' {l} - {i}' for l, i in zip(self.all_letters, self.all_items)])
        )

    def total_nutrition(self):
        ret = 0
        for item in self:
            if item.is_food():
                ret += item.object.nutrition * item.count
        return ret

    def free_slots(self):
        is_coin = any((isinstance(item, O.Coin) for item in self))
        return 52 + is_coin - len(self.all_items)

    def on_panic(self):
        self._previous_inv_strs = None
        self._clear()

    def update(self, force=False):
        if force:
            self._recheck_containers = True

        if force or self._previous_inv_strs is None or \
                (self.agent.last_observation['inv_strs'] != self._previous_inv_strs).any():
            self._clear()
            self._previous_inv_strs = self.agent.last_observation['inv_strs']
            previous_inv_strs = self._previous_inv_strs

            # For some reasons sometime the inventory entries in last_observation may be duplicated
            iterable = set()
            for item_name, category, glyph, letter in zip(
                    self.agent.last_observation['inv_strs'],
                    self.agent.last_observation['inv_oclasses'],
                    self.agent.last_observation['inv_glyphs'],
                    self.agent.last_observation['inv_letters']):
                item_name = bytes(item_name).decode().strip('\0')
                letter = chr(letter)
                if not item_name:
                    continue
                iterable.add((item_name, category, glyph, letter))
            iterable = sorted(iterable, key=lambda x: x[-1])

            assert len(iterable) == len(set(map(lambda x: x[-1], iterable))), \
                   'letters in inventory are not unique'

            for item_name, category, glyph, letter in iterable:
                item = self.agent.inventory.item_manager.get_item_from_text(item_name, category=category,
                        glyph=glyph if not nh.glyph_is_body(glyph) and not nh.glyph_is_statue(glyph) else None,
                        position=None)

                self.all_items.append(item)
                self.all_letters.append(letter)

                if item.equipped:
                    for types, sub, name in [
                        ((O.Weapon, O.WepTool), None,         'main_hand'),
                        (O.Armor,               O.ARM_SHIELD, 'off_hand'), # TODO: twoweapon support
                        (O.Armor,               O.ARM_SUIT,   'suit'),
                        (O.Armor,               O.ARM_HELM,   'helm'),
                        (O.Armor,               O.ARM_GLOVES, 'gloves'),
                        (O.Armor,               O.ARM_BOOTS,  'boots'),
                        (O.Armor,               O.ARM_CLOAK,  'cloak'),
                        (O.Armor,               O.ARM_SHIRT,  'shirt'),
                    ]:
                        if isinstance(item.objs[0], types) and (sub is None or sub == item.objs[0].sub):
                            assert getattr(self, name) is None, ((name, getattr(self, name), item), str(self), iterable)
                            setattr(self, name, item)
                            break

                if item.is_possible_container() or (item.is_container() and self._recheck_containers):
                    self.agent.inventory.check_container_content(item)

                if (self.agent.last_observation['inv_strs'] != previous_inv_strs).any():
                    self.update()
                    return

                self.total_weight += item.weight()
                # weight is sometimes unambiguous for unidentified items. All exceptions:
                # {'helmet': 30, 'helm of brilliance': 50, 'helm of opposite alignment': 50, 'helm of telepathy': 50}
                # {'leather gloves': 10, 'gauntlets of fumbling': 10, 'gauntlets of power': 30, 'gauntlets of dexterity': 10}
                # {'speed boots': 20, 'water walking boots': 15, 'jumping boots': 20, 'elven boots': 15, 'fumble boots': 20, 'levitation boots': 15}
                # {'luckstone': 10, 'loadstone': 500, 'touchstone': 10, 'flint': 10}

            self._recheck_containers = False

    def get_letter(self, item):
        assert item in self.all_items, (item, self.all_items)
        return self.all_letters[self.all_items.index(item)]


class ItemPriorityBase:
    def _split(self, items, forced_items, weight_capacity):
        '''
        returns a dict (container_item or None for inventory) ->
                       (list of counts to take corresponding to `items`)

        Lack of the container in the dict means "don't change the content except for
        items wanted by other containers"

        Order of `items` matters. First items are more important.
        Otherwise the agent will drop and pickup items repeatedly.

        The function should motonic (i.e. removing an item from the argument,
        shouldn't decrease counts of other items). Otherwise the agent may
        go to the item, don't take it, and repeat infinitely

        weight capacity can be exceeded. It's only a hint what the agent wants
        '''
        raise NotImplementedError()

    def split(self, items, forced_items, weight_capacity):
        ret = self._split(items, forced_items, weight_capacity)
        assert None in ret
        counts = np.array(list(ret.values())).sum(0)
        assert all((0 <= count <= item.count for count, item in zip(counts, items)))
        assert all((0 <= c <= item.count for cs in ret.values() for c, item in zip(cs, items)))
        assert all((item not in ret or item.is_container() for item in items))
        assert all((item not in ret or ret[item][i] == 0 for i, item in enumerate(items)))
        return ret


class Inventory:
    _name_to_category = {
        'Amulets': nh.AMULET_CLASS,
        'Armor': nh.ARMOR_CLASS,
        'Comestibles': nh.FOOD_CLASS,
        'Coins': nh.COIN_CLASS,
        'Gems/Stones': nh.GEM_CLASS,
        'Potions': nh.POTION_CLASS,
        'Rings': nh.RING_CLASS,
        'Scrolls': nh.SCROLL_CLASS,
        'Spellbooks': nh.SPBOOK_CLASS,
        'Tools': nh.TOOL_CLASS,
        'Weapons': nh.WEAPON_CLASS,
        'Wands': nh.WAND_CLASS,
        'Boulders/Statues': nh.ROCK_CLASS,
        'Chains': nh.CHAIN_CLASS,
        'Iron balls': nh.BALL_CLASS,
    }

    def __init__(self, agent):
        self.agent = agent
        self.item_manager = ItemManager(self.agent)
        self.items = InventoryItems(self.agent)

        self._previous_blstats = None
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None

        self.skip_engrave_counter = 0

    def on_panic(self):
        self.items_below_me = None
        self.letters_below_me = None
        self.engraving_below_me = None
        self._previous_blstats = None

        self.item_manager.on_panic()
        self.items.on_panic()

    def update(self):
        self.item_manager.update()
        self.items.update()

        if self._previous_blstats is None or \
                (self._previous_blstats.y, self._previous_blstats.x, \
                 self._previous_blstats.level_number, self._previous_blstats.dungeon_number) != \
                (self.agent.blstats.y, self.agent.blstats.x, \
                 self.agent.blstats.level_number, self.agent.blstats.dungeon_number) or \
                (self.engraving_below_me is None or self.engraving_below_me.lower() == 'elbereth'):
            assume_appropriate_message = self._previous_blstats is not None and not self.engraving_below_me

            self._previous_blstats = self.agent.blstats
            self.items_below_me = None
            self.letters_below_me = None
            self.engraving_below_me = None

            self.get_items_below_me(assume_appropriate_message=assume_appropriate_message)

        assert self.items_below_me is not None and self.letters_below_me is not None and self.engraving_below_me is not None

    @contextlib.contextmanager
    def panic_if_items_below_me_change(self):
        old_items_below_me = self.items_below_me
        old_letters_below_me = self.letters_below_me

        def f(self):
            if (
                    [(l, i.text) for i, l in zip(old_items_below_me, old_letters_below_me)] !=
                    [(l, i.text) for i, l in zip(self.items_below_me, self.letters_below_me)]
            ):
                raise AgentPanic('items below me changed')

        fun = partial(f, self)

        self.agent.on_update.append(fun)

        try:
            yield
        finally:
            assert fun in self.agent.on_update
            self.agent.on_update.pop(self.agent.on_update.index(fun))

    ####### ACTIONS

    def wield(self, item, smart=True):
        if smart:
            if item is not None:
                item = self.move_to_inventory(item)

        if item is None: # fists
            letter = '-'
        else:
            letter = self.items.get_letter(item)

        if item is not None and item.equipped:
            return True

        if self.agent.character.prop.polymorph:
            # TODO: depends on kind of a monster
            return False

        if (self.items.main_hand is not None and self.items.main_hand.status == Item.CURSED) or \
                (item is not None and item.objs[0].bi and self.items.off_hand is not None):
            return False

        with self.agent.atom_operation():
            self.agent.step(A.Command.WIELD)
            if "Don't be ridiculous" in self.agent.message:
                return False
            assert 'What do you want to wield' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            if 'You cannot wield a two-handed sword while wearing a shield.' in self.agent.message or \
                    'You cannot wield a two-handed weapon while wearing a shield.' in self.agent.message or \
                    ' welded to your hand' in self.agent.message:
                return False
            assert re.search(r'(You secure the tether\.  )?([a-zA-z] - |welds?( itself| themselves| ) to|'
                             r'You are already wielding that|You are empty handed|You are already empty handed)', \
                             self.agent.message), (self.agent.message, self.agent.popup)

        return True

    def wear(self, item, smart=True):
        assert item is not None

        if smart:
            item = self.move_to_inventory(item)
            # TODO: smart should be more than that (taking off the armor for shirts, etc)
        letter = self.items.get_letter(item)

        if item.equipped:
            return True

        for i in self.items:
            assert not isinstance(i, O.Armor) or i.sub != item.sub or not i.equipped, (i, item)

        with self.agent.atom_operation():
            self.agent.step(A.Command.WEAR)
            if "Don't even bother." in self.agent.message:
                return False
            assert 'What do you want to wear?' in self.agent.message, self.agent.message
            self.agent.type_text(letter)
            assert 'You finish your dressing maneuver.' in self.agent.message or \
                   'You are now wearing ' in self.agent.message or \
                   'Your foot is trapped!' in self.agent.message, self.agent.message

        return True

    def takeoff(self, item):
        # TODO: smart

        assert item is not None and item.equipped, item
        letter = self.items.get_letter(item)
        assert item.status != Item.CURSED, item

        equipped_armors = [i for i in self.items if i.is_armor() and i.equipped]
        assert item in equipped_armors

        with self.agent.atom_operation():
            self.agent.step(A.Command.TAKEOFF)

            is_take_off_message = lambda: \
                    'You finish taking off ' in self.agent.message or \
                    'You were wearing ' in self.agent.message or \
                    'You feel that monsters no longer have difficulty pinpointing your location.' in self.agent.message

            if len(equipped_armors) > 1:
                if is_take_off_message():
                    raise AgentPanic('env did not ask for the item to takeoff')
                assert 'What do you want to take off?' in self.agent.message, self.agent.message
                self.agent.type_text(letter)
            if 'It is cursed.' in self.agent.message or 'They are cursed.' in self.agent.message:
                return False
            assert is_take_off_message(), self.agent.message

        return True

    def use_container(self, container, items_to_put, items_to_take, items_to_put_counts=None, items_to_take_counts=None):
        assert container in self.items.all_items or container in self.items_below_me
        assert all((item in self.items.all_items for item in items_to_put))
        assert all((item in container.content.items for item in items_to_take))
        assert container.is_container()
        assert len(items_to_take) - len(items_to_put) <= self.items.free_slots()  # TODO: take counts into consideration
        assert not container.content.locked, container

        def gen():
            if ' vanished!' in self.agent.message:
                self.item_manager.container_contents.pop(container.container_id)
                raise AgentPanic('some items from the container vanished')
            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '
            assert 'You have no free hand.' not in self.agent.single_message, 'TODO: handle it'
            assert 'Do what with ' in self.agent.single_popup[0]
            if items_to_put and items_to_take:
                yield 'r'
            elif items_to_put and not items_to_take:
                yield 'i'
            elif not items_to_put and items_to_take:
                yield 'o'
            else:
                assert 0
            if items_to_put:
                if 'Put in what type of objects?' in self.agent.single_popup[0]:
                    yield from 'a\r'
                assert 'Put in what?' in self.agent.single_popup[0], (self.agent.single_message, self.agent.single_popup)
                yield from self._select_items_in_popup(items_to_put, items_to_put_counts)
            if items_to_take:
                while not self.agent.single_popup or self.agent.single_popup[0] not in ['Take out what type of objects?', 'Take out what?']:
                    assert ' inside, you are blasted by a ' not in self.agent.message, self.agent.message
                    assert self.agent.single_message or self.agent.single_popup, (self.agent.message, self.agent.popup)
                    yield ' '
                if self.agent.single_popup[0] == 'Take out what type of objects?':
                    yield from 'a\r'
                assert 'Take out what?' in self.agent.single_popup[0]
                yield from self._select_items_in_popup(items_to_take, items_to_take_counts)

                if self.agent._observation['misc'][2]:
                    yield ' '
                while 'You have ' in self.agent.single_message and ' removing ' in self.agent.single_message and \
                        'Continue? [ynq] (q)' in self.agent.single_message:
                    yield 'y'

        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in check_container_content
            if container in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                assert "You can't do that while carrying so much stuff." not in self.agent.message, self.agent.message
                self.agent.step(self.items.get_letter(container), gen())
            elif container in self.items_below_me:
                self.agent.step(A.Command.LOOT)
                while True:
                    assert 'Loot which containers?' not in self.agent.popup, self.agent.popup
                    assert 'Loot in what direction?' not in self.agent.message
                    if "You don't find anything here to loot." in self.agent.message:
                        raise AgentPanic('no container to loot')
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                            position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x))
                    if it.container_id == container.container_id:
                        break
                    self.agent.step('n')

                self.agent.step('y', gen())
            else:
                assert 0

        for item in chain(self.items.all_items, self.items_below_me):
            if item.is_container() and item.container_id == container.container_id:
                self.check_container_content(item)

    def check_container_content(self, item):
        assert item.is_possible_container() or item.is_container()
        assert item in self.items.all_items or item in self.items_below_me

        is_bag_of_tricks = False
        if item.content is not None:
            content = item.content
            content.reset()
        else:
            content = ContainerContent()

        def gen():
            nonlocal content, is_bag_of_tricks

            if 'You carefully open ' in self.agent.single_message or 'You open ' in self.agent.single_message:
                yield ' '

            if 'It develops a huge set of teeth and bites you!' in self.agent.single_message:
                is_bag_of_tricks = True
                return

            if 'Hmmm, it turns out to be locked.' in self.agent.single_message or 'It is locked.' in self.agent.single_message:
                content.locked = True
                yield A.Command.ESC
                return

            if check_if_triggered_container_trap(self.agent.single_message):
                self.agent.stats_logger.log_event('triggered_undetected_trap')
                raise AgentPanic('triggered trap while looting')

            if 'You have no hands!' in self.agent.single_message or \
                    'You have no free hand.' in self.agent.single_message:
                return

            if ' vanished!' in self.agent.message:
                raise AgentPanic('some items from the container vanished')

            if 'cat' in self.agent.message and ' inside the box is ' in self.agent.message:
                raise AgentPanic('encountered a cat in a box')

            assert self.agent.single_popup, (self.agent.single_message)
            if '\no - ' not in '\n'.join(self.agent.single_popup):
                # ':' sometimes doesn't display items correctly if there's >= 22 items (the first page isn't shown)
                yield ':'
                if ' is empty' in self.agent.single_message:
                    return
                # if self.agent.single_popup and 'Contents of ' in self.agent.single_popup[0]:
                #     for text in self.agent.single_popup[1:]:
                #         if not text:
                #             continue
                #         content.items.append(self.item_manager.get_item_from_text(text, position=None))
                #     return
                assert 0, (self.agent.single_message, self.agent.single_popup)

            yield from 'o'
            if ' is empty' in self.agent.single_message and not self.agent.single_popup:
                return
            if self.agent.single_popup and self.agent.single_popup[0] == 'Take out what type of objects?':
                yield from 'a\r'
            if self.agent.single_popup and 'Take out what?' in self.agent.single_popup[0]:
                category = None
                while self.agent._observation['misc'][2]:
                    yield ' '
                assert self.agent.popup.count('Take out what?') == 1, self.agent.popup
                for text in self.agent.popup[self.agent.popup.index('Take out what?') + 1:]:
                    if not text:
                        continue
                    if text in self._name_to_category:
                        category = self._name_to_category[text]
                        continue
                    assert category is not None
                    assert text[1:4] == ' - '
                    text = text[4:]
                    content.items.append(self.item_manager.get_item_from_text(text, category=category, position=None))
                return

            assert 0, (self.agent.single_message, self.agent.single_popup)

        with self.agent.atom_operation():
            # TODO: refactor: the same fragment is in use_container
            if item in self.items.all_items:
                self.agent.step(A.Command.APPLY)
                if "You can't do that while carrying so much stuff." in self.agent.message:
                    return  # TODO: is not changing the content in this case a good way to handle this?
                self.agent.step(self.items.get_letter(item), gen())
                if 'You have no hands!' in self.agent.message:
                    return
            else:
                self.agent.step(A.Command.LOOT)
                while True:
                    if "You don't find anything here to loot." in self.agent.message:
                        raise AgentPanic('no container below me')
                    assert 'Loot which containers?' not in self.agent.popup, self.agent.popup
                    assert 'There is ' in self.agent.message and ', loot it?' in self.agent.message, self.agent.message
                    r = re.findall(r'There is ([a-zA-z0-9# ]+) here\, loot it\? \[ynq\] \(q\)', self.agent.message)
                    assert len(r) == 1, self.agent.message
                    text = r[0]
                    it = self.item_manager.get_item_from_text(text,
                            position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x))
                    if (item.container_id is not None and it.container_id == item.container_id) or \
                            (item.container_id is None and item.text == it.text):
                        break
                    self.agent.step('n')
                self.agent.step('y', gen())

            if is_bag_of_tricks:
                assert item.content is None
                raise AgentPanic('bag of tricks bites')

            if item in self.items.all_items and item.comment != item.container_id:
                self.call_item(item, item.container_id)

            if item.content is None:
                assert item.container_id is not None
                assert item.container_id not in self.item_manager.container_contents
                self.item_manager.container_contents[item.container_id] = content
                item.content = content

            # TODO: make it more elegant
            if len(item.glyphs) == 1 and item.glyphs[0] not in self.item_manager._is_not_bag_of_tricks:
                self.item_manager._is_not_bag_of_tricks.add(item.glyphs[0])
                self.item_manager.update_possible_objects(item)

    def _select_items_in_popup(self, items, counts=None):
        assert counts is None or len(counts) == len(items)
        items = list(items)
        while 1:
            if not self.agent.single_popup:
                raise AgentPanic('no popup, but some items were not selected yet')
            for line_i in range(len(self.agent.single_popup)):
                line = self.agent.single_popup[line_i]
                if line[1:4] != ' - ':
                    continue

                for item in items:
                    if item.text != line[4:]:
                        continue

                    i = items.index(item)
                    letter = line[0]

                    if counts is not None and counts[i] != item.count:
                        yield from str(counts[i])
                    yield letter

                    items.pop(i)
                    if counts is not None:
                        count = counts.pop(i)
                    else:
                        count = None
                    break

                if not items:
                    yield '\r'
                    return

            yield ' '
        assert not items

    def get_items_below_me(self, assume_appropriate_message=False):
        with self.agent.panic_if_position_changes():
            with self.agent.atom_operation():
                if not assume_appropriate_message:
                    self.agent.step(A.Command.LOOK)
                elif 'Things that are here:' in self.agent.popup or \
                        re.search('There are (several|many) objects here\.', self.agent.message):
                    # LOOK is necessary even when 'Things that are here' popup is present for some very rare cases
                    self.agent.step(A.Command.LOOK)

                if 'Something is ' in self.agent.message and 'You read: "' in self.agent.message:
                    index = self.agent.message.index('You read: "') + len('You read: "')
                    assert '"' in self.agent.message[index:]
                    engraving = self.agent.message[index : index + self.agent.message[index:].index('"')]
                    self.engraving_below_me = engraving
                else:
                    self.engraving_below_me = ''

                if 'Things that are here:' not in self.agent.popup and 'There is ' not in '\n'.join(self.agent.popup):
                    if 'You see no objects here.' in self.agent.message:
                        items = []
                        letters = []
                    elif 'You see here ' in self.agent.message:
                        item_str = self.agent.message[self.agent.message.index('You see here ') + len('You see here '):]
                        item_str = item_str[:item_str.index('.')]
                        items = [self.item_manager.get_item_from_text(item_str,
                            position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x))]
                        letters = [None]
                    else:
                        items = []
                        letters = []
                else:
                    self.agent.step(A.Command.PICKUP)  # FIXME: parse LOOK output, add this fragment to pickup method
                    if 'Pick up what?' not in self.agent.popup:
                        if 'You cannot reach the bottom of the pit.' in self.agent.message or \
                                'You cannot reach the bottom of the abyss.' in self.agent.message or \
                                'You cannot reach the floor.' in self.agent.message or \
                                'There is nothing here to pick up.' in self.agent.message or \
                                ' solidly fixed to the floor.' in self.agent.message or \
                                'You read:' in self.agent.message or \
                                "You don't see anything in here to pick up." in self.agent.message or \
                                'You cannot reach the ground.' in self.agent.message or \
                                "You don't feel anything in here to pick up." in self.agent.message:
                            items = []
                            letters = []
                        else:
                            assert 0, (self.agent.message, self.agent.popup)
                    else:
                        lines = self.agent.popup[self.agent.popup.index('Pick up what?') + 1:]
                        category = None
                        items = []
                        letters = []
                        for line in lines:
                            if line in self._name_to_category:
                                category = self._name_to_category[line]
                                continue
                            assert line[1:4] == ' - ', line
                            letter, line = line[0], line[4:]
                            letters.append(letter)
                            items.append(self.item_manager.get_item_from_text(line, category,
                                position=(*self.agent.current_level().key(), self.agent.blstats.y, self.agent.blstats.x)))

                self.items_below_me = items
                self.letters_below_me = letters
                return items

    def pickup(self, items, counts=None):
        # TODO: if polyphormed, sometimes 'You are physically incapable of picking anything up.'
        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert len(items) > 0
        assert all(map(lambda item: item in self.items_below_me, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.letters_below_me[self.items_below_me.index(item)] for item in items]
        screens = [max(self.letters_below_me[:self.items_below_me.index(item) + 1].count('a') - 1, 0) for item in items]

        with self.panic_if_items_below_me_change():
            self.get_items_below_me()

        one_item = len(self.items_below_me) == 1
        with self.agent.atom_operation():
            if one_item:
                assert all((s in [0, None] for s in screens))
                self.agent.step(A.Command.PICKUP)
                drop_count = items[0].count - counts[0]
            else:
                text = ' '.join((
                    ''.join([(str(count) if item.count != count else '') + letter
                             for letter, item, count, screen in zip(letters, items, counts, screens)
                             if count != 0 and screen == current_screen])
                    for current_screen in range(max(screens) + 1)))
                self.agent.step(A.Command.PICKUP, iter(list(text) + [A.MiscAction.MORE]))

            while re.search('You have [a-z ]+ lifting ', self.agent.message) and \
                    'Continue?' in self.agent.message:
                self.agent.type_text('y')
            if one_item and drop_count:
                letter = re.search(r'([a-zA-Z$]) - ', self.agent.message)
                assert letter is not None, self.agent.message
                letter = letter[1]

        if one_item and drop_count:
            self.drop(self.items.all_items[self.items.all_letters.index(letter)], drop_count, smart=False)

        self.get_items_below_me()

        return True

    def drop(self, items, counts=None, smart=True):
        if smart:
            items = self.move_to_inventory(items)

        if isinstance(items, Item):
            items = [items]
            if counts is not None:
                counts = [counts]
        if counts is None:
            counts = [i.count for i in items]
        assert all(map(lambda x: isinstance(x, (int, np.int32, np.int64)), counts)), list(map(type, counts))
        assert len(items) > 0
        assert all(map(lambda item: item in self.items.all_items, items))
        assert len(counts) == len(items)
        assert sum(counts) > 0 and all((0 <= c <= i.count for c, i in zip(counts, items)))

        letters = [self.items.all_letters[self.items.all_items.index(item)] for item in items]
        texts_to_type = [(str(count) if item.count != count else '') + letter
                         for letter, item, count in zip(letters, items, counts) if count != 0]

        if all((not i.can_be_dropped_from_inventory() for i in items)):
            return False

        def key_gen():
            if 'Drop what type of items?' in '\n'.join(self.agent.single_popup):
                yield 'a'
                yield A.MiscAction.MORE
            assert 'What would you like to drop?' in '\n'.join(self.agent.single_popup), \
                   (self.agent.single_message, self.agent.single_popup)
            i = 0
            while texts_to_type:
                for text in list(texts_to_type):
                    letter = text[-1]
                    if f'{letter} - ' in '\n'.join(self.agent.single_popup):
                        yield from text
                        texts_to_type.remove(text)

                if texts_to_type:
                    yield A.TextCharacters.SPACE
                    i += 1

                assert i < 100, ('infinite loop', texts_to_type, self.agent.message)
            yield A.MiscAction.MORE

        with self.agent.atom_operation():
            self.agent.step(A.Command.DROPTYPE, key_gen())
        self.get_items_below_me()

        return True

    def move_to_inventory(self, items):
        # all items in self.items will be updated!

        if not isinstance(items, list):
            is_list = False
            items = [items]
        else:
            is_list = True

        moved_items = {item for item in items if item in self.items.all_items}

        if len(moved_items) != len(items):
            with self.agent.atom_operation():
                its = list(filter(lambda i: i in self.items_below_me, items))
                if its:
                    moved_items = moved_items.union(its)
                    self.pickup(its)
                for container in chain(self.items_below_me, self.items):
                    if container.is_container():
                        its = list(filter(lambda i: i in container.content.items, items))
                        if its:
                            moved_items = moved_items.union(its)
                            self.use_container(container, items_to_take=its, items_to_put=[])

                assert moved_items == set(items), ('TODO: nested containers', moved_items, items)

            # TODO: HACK
            self.agent.last_observation = self.agent.last_observation.copy()
            for key in ['inv_strs', 'inv_oclasses', 'inv_glyphs', 'inv_letters']:
                self.agent.last_observation[key] = self.agent._observation[key].copy()
            self.items.update(force=True)

            ret = []
            for item in items:
                ret.append(find_equivalent_item(item, filter(lambda i: i not in ret, self.items.all_items)))
        else:
            ret = items

        if not is_list:
            assert len(ret) == 1
            return ret[0]
        return ret

    def call_item(self, item, name):
        assert item in self.items.all_items, item
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            self.agent.step(A.Command.CALL, iter(f'i{letter}#{name}\r'))
        return True

    def quaff(self, item, smart=True):
        return self.eat(item, quaff=True, smart=smart)

    def eat(self, item, quaff=False, smart=True):
        if smart:
            if not quaff and item in self.items_below_me:
                with self.agent.atom_operation():
                    self.agent.step(A.Command.EAT)
                    while '; eat it? [ynq]' in self.agent.message or \
                            '; eat one? [ynq]' in self.agent.message:
                        if f'{item.text} here; eat it? [ynq]' in self.agent.message or \
                                f'{item.text} here; eat one? [ynq]' in self.agent.message:
                            self.agent.type_text('y')
                            return True
                        self.agent.type_text('n')
                    # if "What do you want to eat?" in self.agent.message or \
                    #         "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no such food is lying here')
                    assert 0, self.agent.message

            # TODO: eat directly from ground if possible
            item = self.move_to_inventory(item)

        assert item in self.items.all_items, item or item in self.items_below_me
        letter = self.items.get_letter(item)
        with self.agent.atom_operation():
            if quaff:
                def text_gen():
                    if self.agent.message.startswith('Drink from the fountain?'):
                        yield 'n'
                self.agent.step(A.Command.QUAFF, text_gen())
            else:
                self.agent.step(A.Command.EAT)
            if item in self.items.all_items:
                while re.search('There (is|are)[a-zA-Z0-9- ]* here; eat (it|one)\?', self.agent.message):
                    self.agent.type_text('n')
                self.agent.type_text(letter)
                return True

            elif item in self.items_below_me:
                while ' eat it? [ynq]' in self.agent.message or \
                        ' eat one? [ynq]' in self.agent.message:
                    if item.text in self.agent.message:
                        self.type_text('y')
                        return True
                if "What do you want to eat?" in self.agent.message or \
                        "You don't have anything to eat." in self.agent.message:
                    raise AgentPanic('no food is lying here')

                assert 0, self.agent.message

        assert 0

    ######## STRATEGIES helpers

    def get_best_melee_weapon(self, items=None, *, return_dps=False, allow_unknown_status=False):
        if self.agent.character.role == Character.MONK:
            return None

        if items is None:
            items = self.items
        # select the best
        best_item = None
        best_dps = utils.calc_dps(*self.agent.character.get_melee_bonus(None, large_monster=False))
        for item in flatten_items(items):
            if item.is_weapon() and \
                    (item.status in [Item.UNCURSED, Item.BLESSED] or
                     (allow_unknown_status and item.status == Item.UNKNOWN)):
                to_hit, dmg = self.agent.character.get_melee_bonus(item, large_monster=False)
                dps = utils.calc_dps(to_hit, dmg)
                # dps = item.get_dps(large_monster=False)  # TODO: what about monster size
                if best_dps < dps:
                    best_dps = dps
                    best_item = item
        if return_dps:
            return best_item, best_dps
        return best_item

    def get_ranged_combinations(self, items=None, throwing=True, allow_best_melee=False, allow_wielded_melee=False,
                                allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        items = flatten_items(items)
        launchers = [i for i in items if i.is_launcher()]
        ammo_list = [i for i in items if i.is_fired_projectile()]
        valid_combinations = []

        # TODO: should this condition be used here
        if any(l.equipped and l.status == Item.CURSED for l in launchers):
            launchers = [l for l in launchers if l.equipped]

        for launcher in launchers:
            for ammo in ammo_list + additional_ammo:
                if ammo.is_fired_projectile(launcher):
                    if launcher.status in [Item.UNCURSED, Item.BLESSED] or \
                            (allow_unknown_status and launcher.status == Item.UNKNOWN):
                        valid_combinations.append((launcher, ammo))

        if throwing:
            best_melee_weapon = None
            if not allow_best_melee:
                best_melee_weapon = self.get_best_melee_weapon()
            wielded_melee_weapon = None
            if not allow_wielded_melee:
                wielded_melee_weapon = self.items.main_hand
            valid_combinations.extend([(None, i) for i in items
                                       if i.is_thrown_projectile()
                                       and i != best_melee_weapon and i != wielded_melee_weapon])

        return valid_combinations

    def get_best_ranged_set(self, items=None, *, throwing=True, allow_best_melee=False,
                            allow_wielded_melee=False,
                            return_dps=False, allow_unknown_status=False, additional_ammo=[]):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_launcher, best_ammo = None, None
        best_dps = -float('inf')
        for launcher, ammo in self.get_ranged_combinations(items, throwing, allow_best_melee, allow_wielded_melee,
                                                           allow_unknown_status, additional_ammo):
            to_hit, dmg = self.agent.character.get_ranged_bonus(launcher, ammo)
            dps = utils.calc_dps(to_hit, dmg)
            if dps > best_dps:
                best_launcher, best_ammo, best_dps = launcher, ammo, dps
        if return_dps:
            return best_launcher, best_ammo, best_dps
        return best_launcher, best_ammo

    def get_best_armorset(self, items=None, *, return_ac=False, allow_unknown_status=False):
        if items is None:
            items = self.items
        items = flatten_items(items)

        best_items = [None] * O.ARM_NUM
        best_ac = [None] * O.ARM_NUM
        for item in items:
            if not item.is_armor() or not item.is_unambiguous():
                continue

            # TODO: consider other always allowed items than dragon hide
            is_dragonscale_armor = item.object.metal == O.DRAGON_HIDE

            allowed_statuses = [Item.UNCURSED, Item.BLESSED] + ([Item.UNKNOWN] if allow_unknown_status else [])
            if item.status not in allowed_statuses and not is_dragonscale_armor:
                continue

            slot = item.object.sub
            ac = item.get_ac()

            if self.agent.character.role == Character.MONK and slot == O.ARM_SUIT:
                continue

            if best_ac[slot] is None or best_ac[slot] > ac:
                best_ac[slot] = ac
                best_items[slot] = item

        if return_ac:
            return best_items, best_ac
        return best_items

    ######## LOW-LEVEL STRATEGIES

    def gather_items(self):
        return (
            self.pickup_and_drop_items()
            .before(self.check_containers())
            .before(self.wear_best_stuff())
            .before(self.wand_engrave_identify())
            .before(self.go_to_unchecked_containers())
            .before(self.check_items()
                    .before(self.go_to_item_to_pickup()).repeat().every(5)
                    .preempt(self.agent, [
                        self.pickup_and_drop_items(),
                        self.check_containers(),
                    ])).repeat()
        )

    @utils.debug_log('inventory.arrange_items')
    @Strategy.wrap
    def arrange_items(self):
        yielded = False

        if self.agent.character.prop.polymorph:
            # TODO: only handless
            yield False

        while 1:
            items_below_me = list(filter(lambda i: i.shop_status == Item.NOT_SHOP, flatten_items(self.items_below_me)))
            forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), flatten_items(self.items)))
            assert all((item in self.items.all_items for item in forced_items))
            free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(),
                                     flatten_items(sorted(self.items, key=lambda x: x.text))))
            all_items = free_items + items_below_me

            item_split = self.agent.global_logic.item_priority.split(
                    all_items, forced_items, self.agent.character.carrying_capacity)

            assert all((container is None or container in self.items_below_me or container in self.items.all_items or \
                        (sum(item_split[container]) == 0 and not container.content.items)
                        for container in item_split)), 'TODO: nested containers'

            cont = False

            # put into containers
            for container in item_split:
                if container is not None:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if item in self.items.all_items and counts[i] > 0]
                    if not indices:
                        continue
                    if not yielded:
                        yielded = True
                        yield True

                    self.use_container(container, [all_items[i] for i in indices], [],
                                       items_to_put_counts=[counts[i] for i in indices])
                    cont = True
                    break
            if cont:
                continue

            # drop on ground
            counts = item_split[None]
            indices = [i for i, item in enumerate(free_items) if item in self.items.all_items and counts[i] != item.count]
            if indices:
                if not yielded:
                    yielded = True
                    yield True
                assert self.drop([free_items[i] for i in indices], [free_items[i].count - counts[i] for i in indices], smart=False)
                continue

            # take from container
            for container in all_items:
                if not container.is_container():
                    continue

                if container in item_split:
                    counts = item_split[container]
                    indices = [i for i, item in enumerate(all_items) if item in container.content.items and counts[i] != item.count]
                    items_to_take_counts = [all_items[i].count - counts[i] for i in indices]
                else:
                    counts = np.array(list(item_split.values())).sum(0)
                    indices = [i for i, item in enumerate(all_items) if item in container.content.items and counts[i] != 0]
                    items_to_take_counts = [counts[i] for i in indices]

                if not indices:
                    continue
                if not yielded:
                    yielded = True
                    yield True

                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]

                self.use_container(container, [], [all_items[i] for i in indices],
                                   items_to_take_counts=items_to_take_counts)
                cont = True
                break
            if cont:
                continue

            # pick up from ground
            to_pickup = np.array([counts[len(free_items):] for counts in item_split.values()]).sum(0)
            assert len(to_pickup) == len(items_below_me)
            indices = [i for i, item in enumerate(items_below_me) if to_pickup[i] > 0 and item in self.items_below_me]
            if len(indices) > 0:
                assert self.items.free_slots() > 0
                indices = indices[:self.items.free_slots()]
                if not yielded:
                    yielded = True
                    yield True
                assert self.pickup([items_below_me[i] for i in indices], [to_pickup[i] for i in indices])
                continue

            break

        for container in item_split:
            for item, count in zip(all_items, item_split[container]):
                assert count == 0 or count == item.count
                assert count == 0 or item in (container.content.items if container is not None else self.items.all_items)

        if not yielded:
            yield False

    def _determine_possible_wands(self, message, item):

        wand_regex = '[a-zA-Z ]+'
        floor_regex = '[a-zA-Z]+'
        mapping = {
            f"The engraving on the {floor_regex} vanishes!": ['cancellation', 'teleportation', 'make invisible'],
            # TODO?: cold,  # (if the existing engraving is a burned one)

            "A few ice cubes drop from the wand.": ['cold'],
            f"The bugs on the {floor_regex} stop moving": ['death', 'sleep'],
            f"This {wand_regex} is a wand of digging!": ['digging'],
            "Gravel flies up from the floor!": ['digging'],
            f"This {wand_regex} is a wand of fire!": ['fire'],
            "Lightning arcs from the wand. You are blinded by the flash!": ['lighting'],
            f"This {wand_regex} is a wand of lightning!": ['lightning'],
            f"The {floor_regex} is riddled by bullet holes!": ['magic missile'],
            f'The engraving now reads:': ['polymorph'],
            f"The bugs on the {floor_regex} slow down!": ['slow monster'],
            f"The bugs on the {floor_regex} speed up!": ['speed monster'],
            "The wand unsuccessfully fights your attempt to write!": ['striking'],

            # activated effects:
            "A lit field surrounds you!": ['light'],
            "You may wish for an object.": ['wishing'],
            "You feel self-knowledgeable...": ['enlightenment']  # TODO: parse the effect
            # TODO: "The wand is too worn out to engrave.": [None],  # wand is exhausted
        }

        for msg, wand_types in mapping.items():
            res = re.findall(msg, message)
            if len(res) > 0:
                assert len(res) == 1
                return [O.from_name(w, nh.WAND_CLASS) for w in wand_types]

        # TODO: "wand is cancelled (x:-1)" ?
        # TODO: "secret door detection self-identifies if secrets are detected" ?

        res = re.findall(f'Your {wand_regex} suddenly explodes!', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return None

        res = re.findall('The wand is too worn out to engrave.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            self.agent.inventory.call_item(item, 'EMPT')
            return None

        res = re.findall(f'{wand_regex} glows, then fades.', self.agent.message)
        if len(res) > 0:
            assert len(res) == 1
            return [p for p in O.possibilities_from_glyph(item.glyphs[0])
                    if p.name not in ['light', 'wishing']]
            # TODO: wiki says this:
            # return [O.from_name('opening', nh.WAND_CLASS),
            #         O.from_name('probing', nh.WAND_CLASS),
            #         O.from_name('undead turning', nh.WAND_CLASS),
            #         O.from_name('nothing', nh.WAND_CLASS),
            #         O.from_name('secret door detection', nh.WAND_CLASS),
            #         ]

        assert 0, message

    @utils.debug_log('inventory.wand_engrave_identify')
    @Strategy.wrap
    def wand_engrave_identify(self):
        if self.agent.character.prop.polymorph:
            yield False  # TODO: only for handless monsters (which cannot write)

        self.skip_engrave_counter -= 1
        if self.agent.character.prop.blind or self.skip_engrave_counter > 0:
            yield False
            return
        yielded = False
        for item in self.agent.inventory.items:
            if not isinstance(item.objs[0], O.Wand):
                continue
            if item.is_unambiguous():
                continue
            if self.agent.current_level().objects[self.agent.blstats.y, self.agent.blstats.x] not in G.FLOOR:
                continue
            if item.glyphs[0] in self.item_manager._already_engraved_glyphs:
                continue
            if len(item.glyphs) > 1:
                continue
            if item.comment == 'EMPT':
                continue

            if not yielded:
                yield True
            yielded = True
            self.skip_engrave_counter = 8

            with self.agent.atom_operation():
                wand_types = self._engrave_single_wand(item)

                if wand_types is None:
                    # there is a problem with engraving on this tile
                    continue

                self.item_manager._glyph_to_possible_wand_types[item.glyphs[0]] = wand_types
                self.item_manager._already_engraved_glyphs.add(item.glyphs[0])
                self.item_manager.possible_objects_from_glyph(item.glyphs[0])

            # uncomment for debugging (stopping when there is a new wand being identified)
            # print(len(self.item_manager.possible_objects_from_glyph(item.glyphs[0])))
            # print(self.item_manager._glyph_to_possible_wand_types)
            # input('==================3')

        if yielded:
            self.agent.inventory.items.update(force=True)

        if not yielded:
            yield False

    def _engrave_single_wand(self, item):
        """ Returns possible objects or None if current tile not suitable for identification."""

        def msg():
            return self.agent.message

        def smsg():
            return self.agent.single_message

        self.agent.step(A.Command.LOOK)
        if msg() != 'You see no objects here.':
            return None
        # if 'written' in msg() or 'engraved' in msg() or 'see' not in msg() or 'read' in msg():
        #     return None

        skip_engraving = [False]


        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield '-'
            # if 'Do you want to add to the current engraving' in smsg():
            #     yield 'q'
            #     assert smsg().strip() == 'Never mind.', smsg()
            #     skip_engraving[0] = True
            #     return
            if smsg().startswith('You wipe out the message that was written'):
                yield ' '
                skip_engraving[0] = True
                return
            if smsg().startswith('You cannot wipe out the message that is burned into the floor here.'):
                skip_engraving[0] = True
                return
            assert smsg().startswith('You write in the dust with your fingertip.'), smsg()
            yield ' '
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield 'x'
            assert smsg().startswith('What do you want to write in the dust here?'), smsg()
            yield '\r'

        for _ in range(5):
            # write 'x' with finger in the dust
            self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

            if skip_engraving[0]:
                assert msg().strip().endswith('Never mind.') \
                       or 'You cannot wipe out the message that is burned into the floor here.' in msg(), msg()
                return None

            # this is usually true, but something unrelated like: "You hear crashing rock." may happen
            # assert msg().strip() in '', msg()

            # check if the written 'x' is visible when looking
            self.agent.step(A.Command.LOOK)
            if 'Something is written here in the dust.' in msg() \
                    and 'You read: "x"' in msg():
                break
            else:
                # this is usually true, but something unrelated like:
                #   "There is a doorway here.  Something is written here in the dust. You read: "4".
                #    You see here a giant rat corpse."
                # may happen
                # assert "You see no objects here" in msg(), msg()
                return None
        else:
            assert 0, msg()

        # try engraving with the wand
        letter = self.agent.inventory.items.get_letter(item)
        possible_wand_types = []

        def action_generator():
            assert smsg().startswith('What do you want to write with?'), smsg()
            yield letter
            if 'Do you want to add to the current engraving' in smsg():
                self.agent.type_text('y')
                # assert 'You add to the writing in the dust with' in smsg(), smsg()
                # self.agent.type_text(' ')
            r = self._determine_possible_wands(smsg(), item)
            if r is not None:
                possible_wand_types.extend(r)
            else:
                # wand exploded
                skip_engraving[0] = True

        self.agent.step(A.Command.ENGRAVE, additional_action_iterator=iter(action_generator()))

        if skip_engraving[0]:
            return None

        if 'Do you want to add to the current engraving' in smsg():
            self.agent.type_text('q')
            assert smsg().strip() == 'Never mind.', smsg()

        return possible_wand_types

    @utils.debug_log('inventory.wear_best_stuff')
    @Strategy.wrap
    def wear_best_stuff(self):
        yielded = False
        while 1:
            best_armorset = self.get_best_armorset()

            # TODO: twoweapon
            for slot, name in [(O.ARM_SHIELD, 'off_hand'), (O.ARM_HELM, 'helm'), (O.ARM_GLOVES, 'gloves'),
                               (O.ARM_BOOTS, 'boots'), (O.ARM_SHIRT, 'shirt'), (O.ARM_SUIT, 'suit'),
                               (O.ARM_CLOAK, 'cloak')]:
                if best_armorset[slot] == getattr(self.items, name) or \
                        (getattr(self.items, name) is not None and getattr(self.items, name).status == Item.CURSED):
                    continue
                additional_cond = True
                if slot == O.ARM_SHIELD:
                    additional_cond &= self.items.main_hand is None or not self.items.main_hand.objs[0].bi
                if slot == O.ARM_GLOVES:
                    additional_cond &= self.items.main_hand is None or self.items.main_hand.status != Item.CURSED
                if slot == O.ARM_SHIRT or slot == O.ARM_SUIT:
                    additional_cond &= self.items.cloak is None or self.items.cloak.status != Item.CURSED
                if slot == O.ARM_SHIRT:
                    additional_cond &= self.items.suit is None or self.items.suit.status != Item.CURSED

                if additional_cond:
                    if not yielded:
                        yielded = True
                        yield True
                    if (slot == O.ARM_SHIRT or slot == O.ARM_SUIT) and self.items.cloak is not None:
                        self.takeoff(self.items.cloak)
                        break
                    if slot == O.ARM_SHIRT and self.items.suit is not None:
                        self.takeoff(self.items.suit)
                        break
                    if getattr(self.items, name) is not None:
                        self.takeoff(getattr(self.items, name))
                        break
                    assert best_armorset[slot] is not None
                    self.wear(best_armorset[slot])
                    break
            else:
                break

        if not yielded:
            yield False

    @utils.debug_log('inventory.check_items')
    @Strategy.wrap
    def check_items(self):
        mask = utils.isin(self.agent.glyphs, G.OBJECTS, G.BODIES, G.STATUES)
        if not mask.any():
            yield False

        dis = self.agent.bfs()

        mask &= self.agent.current_level().item_count == 0
        if not mask.any():
            yield False

        mask &= dis > 0
        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.go_to_unchecked_containers')
    @Strategy.wrap
    def go_to_unchecked_containers(self):
        mask = self.agent.current_level().item_count != 0
        if not mask.any():
            yield False

        dis = self.agent.bfs()
        mask &= dis > 0
        if not mask.any():
            yield False

        for y, x in zip(*mask.nonzero()):
            for item in self.agent.current_level().items[y, x]:
                if not item.is_possible_container():
                    mask[y, x] = False

        if not mask.any():
            yield False
        yield True

        nonzero_y, nonzero_x = (mask & (dis == dis[mask].min())).nonzero()
        i = self.agent.rng.randint(len(nonzero_y))
        target_y, target_x = nonzero_y[i], nonzero_x[i]

        with self.agent.env.debug_tiles(mask, color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.check_containers')
    @Strategy.wrap
    def check_containers(self):
        yielded = False
        for item in self.agent.inventory.items_below_me:
            if item.is_possible_container():
                if not yielded:
                    yielded = True
                    yield True
                if item.is_chest() and not (item.is_unambiguous() and item.object.name == 'ice box'):
                    fail_msg = self.agent.untrap_container_below_me()
                    if fail_msg is not None and check_if_triggered_container_trap(fail_msg):
                        raise AgentPanic('triggered trap while looting')
                self.check_container_content(item)
        if not yielded:
            yield False

    @utils.debug_log('inventory.go_to_item_to_pickup')
    @Strategy.wrap
    def go_to_item_to_pickup(self):
        level = self.agent.current_level()
        dis = self.agent.bfs()

        # TODO: free (no charge) items
        mask = ~level.shop_interior & (dis > 0)
        if not mask.any():
            yield False

        mask[mask] = self.agent.current_level().item_count[mask] != 0

        items = {}
        for y, x in sorted(zip(*mask.nonzero()), key=lambda p: dis[p]):
            for i in level.items[y, x]:
                assert i not in items
                items[i] = (y, x)

        if not items:
            yield False

        items = {i: pos for item, pos in items.items() for i in flatten_items([item])}

        free_items = list(filter(lambda i: i.can_be_dropped_from_inventory(), flatten_items(self.items)))
        forced_items = list(filter(lambda i: not i.can_be_dropped_from_inventory(), flatten_items(self.items)))
        item_split = self.agent.global_logic.item_priority.split(
                free_items + list(items.keys()), forced_items,
                self.agent.character.carrying_capacity)
        counts = np.array(list(item_split.values())).sum(0)

        counts = counts[len(free_items):]
        assert len(counts) == len(items)
        if sum(counts) == 0:
            yield False
        yield True

        for (i, _), c in sorted(zip(items.items(), counts), key=lambda x: dis[x[0][1]]):
            if c != 0:
                target_y, target_x = items[i]
                break
        else:
            assert 0

        with self.agent.env.debug_tiles([(y, x) for _, (y, x) in items.items()], color=(255, 0, 0, 128)):
            self.agent.go_to(target_y, target_x, debug_tiles_args=dict(color=(255, 0, 255), is_path=True))

    @utils.debug_log('inventory.pickup_and_drop_items')
    @Strategy.wrap
    def pickup_and_drop_items(self):
        # TODO: free (no charge) items
        self.item_manager.price_identification()
        if self.agent.current_level().shop_interior[self.agent.blstats.y, self.agent.blstats.x]:
            yield False
        if len(self.items_below_me) == 0:
            yield False

        yield from self.arrange_items().strategy()
