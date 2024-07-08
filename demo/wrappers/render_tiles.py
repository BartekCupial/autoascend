import re
from collections import namedtuple

import cv2
import gym
import numpy as np

BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


HISTORY_SIZE = 13
FONT_SIZE = 32
RENDERS_HISTORY_SIZE = 128


def _draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _put_text(img, text, pos, scale=FONT_SIZE / 32, thickness=1, color=(255, 255, 0), console=False):
    # TODO: figure out how exactly opencv anchors the text
    pos = (pos[0] + FONT_SIZE // 2, pos[1] + FONT_SIZE // 2 + 8)

    font = cv2.FONT_HERSHEY_PLAIN  # Monospaced font
    scale *= 2  # Adjust scale for better visibility in console

    return cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _draw_frame(img, color=(90, 90, 90), thickness=3):
    return cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


class RenderTiles(gym.Wrapper):
    def __init__(self, env: gym.Env, tileset_path, tile_size=32):
        super().__init__(env)

        self.tileset = cv2.imread(tileset_path)[..., ::-1]
        if self.tileset is None:
            raise FileNotFoundError(f"Tileset {tileset_path} not found")
        if self.tileset.shape[0] % tile_size != 0 or self.tileset.shape[1] % tile_size != 0:
            raise ValueError("Tileset and tile_size doesn't match modulo")

        h = self.tileset.shape[0] // tile_size
        w = self.tileset.shape[1] // tile_size
        tiles = []
        for y in range(h):
            y *= tile_size
            for x in range(w):
                x *= tile_size
                tiles.append(self.tileset[y : y + tile_size, x : x + tile_size])
        self.tileset = np.array(tiles)
        from glyph2tile import glyph2tile

        self.glyph2tile = np.array(glyph2tile)

        self.frames = []
        # self.output_path = output_path
        self.video_writer = None

        self.action_history = list()
        self.message_history = list()
        self.popup_history = list()

        self._window_name = "NetHackVis"

    def reset(self, **kwargs):
        self.frames = []
        obs = self.env.reset(**kwargs)
        self.render()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.render()

        self.action_history.append(self.env.actions[action].name)
        self.update_message_and_popup_history()

        return obs, reward, done, info

    def render(self, **kwargs):
        glyphs = self.env.last_observation[self.env.unwrapped._glyph_index]

        tiles_idx = self.glyph2tile[glyphs]
        tiles = self.tileset[tiles_idx.reshape(-1)]
        scene_vis = _draw_grid(tiles, glyphs.shape[1])

        self.frames.append(scene_vis)

        topbar = self._draw_topbar(scene_vis.shape[1])
        bottombar = self._draw_bottombar(scene_vis.shape[1])
        rendered = np.concatenate([topbar, scene_vis, bottombar], axis=0)
        image = rendered[..., ::-1]

        ratio = 0.5
        width, height = round(image.shape[1] * ratio), round(image.shape[0] * ratio)

        resized_image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        cv2.imshow(self._window_name, resized_image)
        cv2.waitKey(1)

    def _draw_bottombar(self, width):
        tty_chars = self.env.last_observation[self.env._observation_keys.index("tty_chars")]

        height = FONT_SIZE * len(tty_chars)
        tty = self._draw_tty(tty_chars, width - width // 2, height)
        stats = self._draw_stats(width // 2, height)
        return np.concatenate([tty, stats], axis=1)

    def _draw_tty(self, tty_chars, width, height):
        vis = np.zeros((height, width, 3)).astype(np.uint8)
        for i, line in enumerate(tty_chars):
            txt = "".join([chr(i) for i in line])
            _put_text(vis, txt, (0, i * FONT_SIZE), console=True)
        _draw_frame(vis)
        return vis

    def _draw_stats(self, width, height):
        ret = np.zeros((height, width, 3), dtype=np.uint8)
        blstats = BLStats(*self.env.last_observation[self.env.unwrapped._blstats_index])

        # game info
        i = 0
        txt = [
            f"Dlvl: {blstats.level_number}",
            f"Step: {len(self.action_history)}",
            f"Turn: {blstats.time}",
            f"Score: {blstats.score}",
        ]
        _put_text(ret, " | ".join(txt), (0, i * FONT_SIZE), color=(255, 255, 255))
        i += 3

        # general character info
        txt = [
            f"St:{blstats.strength}",
            f"Dx:{blstats.dexterity}",
            f"Co:{blstats.constitution}",
            f"In:{blstats.intelligence}",
            f"Wi:{blstats.wisdom}",
            f"Ch:{blstats.charisma}",
        ]
        _put_text(ret, " | ".join(txt), (0, i * FONT_SIZE))
        i += 1
        txt = [
            f"HP: {blstats.hitpoints} / {blstats.max_hitpoints}",
            f"LVL: {blstats.experience_level}",
            f"ENERGY: {blstats.energy} / {blstats.max_energy}",
        ]
        hp_ratio = blstats.hitpoints / blstats.max_hitpoints
        hp_color = cv2.applyColorMap(np.array([[130 - int((1 - hp_ratio) * 110)]], dtype=np.uint8), cv2.COLORMAP_TURBO)[
            0, 0
        ]
        _put_text(ret, " | ".join(txt), (0, i * FONT_SIZE), color=tuple(map(int, hp_color)))
        i += 2

        _draw_frame(ret)
        return ret

    def _draw_topbar(self, width):
        actions_vis = self._draw_action_history(np.round(width / 100 * 7).astype(int))
        messages_vis = self._draw_message_history(np.round(width / 100 * 43).astype(int))
        popup_vis = self._draw_popup_history(np.round(width / 100 * 50).astype(int))
        ret = np.concatenate([actions_vis, messages_vis, popup_vis], axis=1)
        assert ret.shape[1] == width
        return ret

    def _draw_action_history(self, width):
        vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.action_history):
                break
            txt = self.action_history[-i - 1]
            if i == 0:
                _put_text(vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(vis)
        return vis

    def _draw_message_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.message_history):
                break
            txt = self.message_history[-i - 1]
            if i == 0:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(messages_vis)
        return messages_vis

    def _draw_popup_history(self, width):
        messages_vis = np.zeros((FONT_SIZE * HISTORY_SIZE, width, 3)).astype(np.uint8)
        for i in range(HISTORY_SIZE):
            if i >= len(self.popup_history):
                break
            txt = "|".join(self.popup_history[-i - 1])
            if i == 0:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(255, 255, 255))
            else:
                _put_text(messages_vis, txt, (0, i * FONT_SIZE), color=(120, 120, 120))
        _draw_frame(messages_vis)
        return messages_vis

    def update_message_and_popup_history(self):
        """Uses MORE action to get full popup and/or message."""
        message = self.env.last_observation[self.env.unwrapped._message_index]
        message = bytes(message).decode().replace("\0", " ").replace("\n", "").strip()
        if message.endswith("--More--"):
            # FIXME: It seems like in this case the environment doesn't expect additional input,
            #        but I'm not 100% sure, so it's too risky to change it, because it could stall everything.
            #        With the current implementation, in the worst case, we'll get "Unknown command ' '".
            message = message[: -len("--More--")]

        assert "\n" not in message and "\r" not in message
        popup = []

        tty_chars = self.env.last_observation[self.env._observation_keys.index("tty_chars")]
        lines = [bytes(line).decode().replace("\0", " ").replace("\n", "") for line in tty_chars]
        marker_pos, marker_type = self._find_marker(lines)

        if marker_pos is None:
            self.message_history.append(message)
            self.popup_history.append(popup)
            return

        pref = ""
        message_lines_count = 0
        if message:
            for i, line in enumerate(lines[: marker_pos[0] + 1]):
                if i == marker_pos[0]:
                    line = line[: marker_pos[1]]
                message_lines_count += 1
                pref += line.strip()

                # I'm not sure when the new line character in broken messages should be a space and when be ignored.
                # '#' character (and others) occasionally occurs at the beginning of the broken line and isn't in
                # the message. Sometimes the message on the screen lacks last '.'.
                def replace_func(x):
                    return "".join((c for c in x if c.isalnum()))

                if replace_func(pref) == replace_func(message):
                    break
            else:
                if marker_pos[0] == 0:
                    elems1 = [s for s in message.split() if s]
                    elems2 = [s for s in pref.split() if s]
                    assert len(elems1) < len(elems2) and elems2[-len(elems1) :] == elems1, (elems1, elems2)
                    return pref, popup, False
                raise ValueError(f"Message:\n{repr(message)}\ndoesn't match the screen:\n{repr(pref)}")

        # cut out popup
        for line in lines[message_lines_count : marker_pos[0]] + [lines[marker_pos[0]][: marker_pos[1]]]:
            line = line[marker_pos[1] :].strip()
            if line:
                popup.append(line)

        self.message_history.append(message)
        self.popup_history.append(popup)

    @staticmethod
    def _find_marker(lines, regex=re.compile(r"(--More--|\(end\)|\(\d+ of \d+\))")):
        """Return (line, column) of markers:
        --More-- | (end) | (X of N)
        """
        if len(regex.findall(" ".join(lines))) > 1:
            raise ValueError("Too many markers")

        result, marker_type = None, None
        for i, line in enumerate(lines):
            res = regex.findall(line)
            if res:
                assert len(res) == 1
                j = line.find(res[0])
                result, marker_type = (i, j), res[0]
                break

        if result is not None and result[1] == 1:
            result = (result[0], 0)  # e.g. for known items view
        return result, marker_type
