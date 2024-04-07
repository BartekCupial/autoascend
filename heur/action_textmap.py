"""
taken from https://github.com/upiterbarg/diff_history/blob/main/action_textmap.py
"""

special_tokens_interaction_history = {
    "action": "<|action|>",
    "observation": "<|observation|>",
}

nle_obs_preqs = {
    "txt_blstats": "statistics",
    "txt_glyphs": "glyphs",
    "txt_message": "message",
    "txt_inventory": "inventory",
    "txt_cursor": "cursor",
    "prev_action_seq": "last keypress",
    "prev_action_seq_kphist": "keypress history",
}

nle_comp_preqs = {
    "action": "<|action|>",
    "strategy": "<|strategy|>",
    "eos": "</s>",
}

nle_action_textmap = {
    "UnsafeActions.HELP": "help",
    "UnsafeActions.PREVMSG": "previous message",
    "CompassDirection.N": "north",
    "CompassDirection.E": "east",
    "CompassDirection.S": "south",
    "CompassDirection.W": "west",
    "MiscDirection.UP": "up",
    "MiscDirection.DOWN": "down",
    "MiscDirection.WAIT": "wait",
    "MiscAction.MORE": "more",
    "Command.EXTCMD": "extcmd",
    "Command.EXTLIST": "extlist",
    "Command.ADJUST": "adjust",
    "Command.ANNOTATE": "annotate",
    "Command.APPLY": "apply",
    "Command.ATTRIBUTES": "attributes",
    "Command.AUTOPICKUP": "autopickup",
    "Command.CALL": "call",
    "Command.CAST": "cast",
    "Command.CHAT": "chat",
    "Command.CLOSE": "close",
    "Command.CONDUCT": "conduct",
    "Command.DIP": "dip",
    "Command.DROP": "drop",
    "Command.DROPTYPE": "droptype",
    "Command.EAT": "eat",
    "Command.ESC": "esc",
    "Command.ENGRAVE": "engrave",
    "Command.ENHANCE": "enhance",
    "Command.FIRE": "fire",
    "Command.FIGHT": "fight",
    "Command.FORCE": "force",
    "Command.GLANCE": "glance",
    "Command.HISTORY": "history",
    "Command.INVENTORY": "inventory",
    "Command.INVENTTYPE": "inventtype",
    "Command.INVOKE": "invoke",
    "Command.JUMP": "jump",
    "Command.KICK": "kick",
    "Command.KNOWN": "known",
    "Command.KNOWNCLASS": "knownclass",
    "Command.LOOK": "look",
    "Command.LOOT": "loot",
    "Command.MONSTER": "monster",
    "Command.MOVE": "move",
    "Command.MOVEFAR": "movefar",
    "Command.OFFER": "offer",
    "Command.OPEN": "open",
    "Command.OPTIONS": "options",
    "Command.OVERVIEW": "wizard where",
    "Command.PAY": "pay",
    "Command.PICKUP": "pickup",
    "Command.PRAY": "pray",
    "Command.PUTON": "puton",
    "Command.QUAFF": "quaff",
    "Command.QUIT": "quit",
    "Command.QUIVER": "quiver",
    "Command.READ": "read",
    "Command.REDRAW": "redraw",
    "Command.REMOVE": "remove",
    "Command.RIDE": "ride",
    "Command.RUB": "rub",
    "Command.RUSH": "rush",
    "Command.RUSH2": "rush2",
    "Command.SAVE": "save",
    "Command.SEARCH": "search",
    "Command.SEEALL": "seeall",
    "Command.SEEAMULET": "seeamulet",
    "Command.SEEARMOR": "seearmor",
    "Command.SEEGOLD": "seegold",
    "Command.SEERINGS": "seerings",
    "Command.SEESPELLS": "seespells",
    "Command.SEETOOLS": "seetools",
    "Command.SEETRAP": "seetrap",
    "Command.SEEWEAPON": "seeweapon",
    "Command.SHELL": "shell",
    "Command.SIT": "sit",
    "Command.SWAP": "swap",
    "Command.TAKEOFF": "takeoff",
    "Command.TAKEOFFALL": "takeoffall",
    "Command.TELEPORT": "teleport",
    "Command.THROW": "throw",
    "Command.TIP": "tip",
    "Command.TRAVEL": "travel",
    "Command.TURN": "turnundead",
    "Command.TWOWEAPON": "twoweapon",
    "Command.UNTRAP": "untrap",
    "Command.VERSION": "version",
    "Command.VERSIONSHORT": "versionshort",
    "Command.WEAR": "wear",
    "Command.WHATDOES": "whatdoes",
    "Command.WHATIS": "whatis",
    "Command.WIELD": "wield",
    "Command.WIPE": "wipe",
    "Command.ZAP": "zap",
    "TextCharacters.MINUS": "minus",
    "TextCharacters.SPACE": "space",
    "TextCharacters.APOS": "apos",
    "TextCharacters.NUM_0": "zero",
    "TextCharacters.NUM_1": "one",
    "TextCharacters.NUM_2": "two",
    "TextCharacters.NUM_3": "three",
    "TextCharacters.NUM_4": "four",
    "TextCharacters.NUM_5": "five",
    "TextCharacters.NUM_6": "six",
    "TextCharacters.NUM_7": "seven",
    "TextCharacters.NUM_8": "eight",
    "TextCharacters.NUM_9": "nine",
    "WizardCommand.WIZDETECT": "wizard detect",
    "WizardCommand.WIZGENESIS": "wizard genesis",
    "WizardCommand.WIZIDENTIFY": "wizard identify",
    "WizardCommand.WIZLEVELPORT": "wizard teleport",
    "WizardCommand.WIZMAP": "wizard map",
    "WizardCommand.WIZWISH": "wizard wish",
    "CompassDirection.NE": "northeast",
    "CompassDirection.SE": "southeast",
    "CompassDirection.SW": "southwest",
    "CompassDirection.NW": "northwest",
    "CompassDirectionLonger.N": "far north",
    "CompassDirectionLonger.E": "far east",
    "CompassDirectionLonger.S": "far south",
    "CompassDirectionLonger.W": "far west",
    "CompassDirectionLonger.NE": "far northeast",
    "CompassDirectionLonger.SE": "far southeast",
    "CompassDirectionLonger.SW": "far southwest",
    "CompassDirectionLonger.NW": "far northwest",
    "TextCharacters.PLUS": "+",
    "TextCharacters.QUOTE": '"',
    "TextCharacters.DOLLAR": "$",
}

all_nle_action_strs = [
    "help",
    "previous message",
    "north",
    "east",
    "south",
    "west",
    "up",
    "down",
    "wait",
    "more",
    "extcmd",
    "extlist",
    "adjust",
    "annotate",
    "apply",
    "attributes",
    "autopickup",
    "call",
    "cast",
    "chat",
    "close",
    "conduct",
    "dip",
    "drop",
    "droptype",
    "eat",
    "esc",
    "engrave",
    "enhance",
    "fire",
    "fight",
    "force",
    "glance",
    "history",
    "inventory",
    "inventtype",
    "invoke",
    "jump",
    "kick",
    "known",
    "knownclass",
    "look",
    "loot",
    "monster",
    "move",
    "movefar",
    "offer",
    "open",
    "options",
    "wizard where",
    "pay",
    "pickup",
    "pray",
    "puton",
    "quaff",
    "quit",
    "quiver",
    "read",
    "redraw",
    "remove",
    "ride",
    "rub",
    "rush",
    "rush2",
    "save",
    "search",
    "seeall",
    "seeamulet",
    "seearmor",
    "seegold",
    "seerings",
    "seespells",
    "seetools",
    "seetrap",
    "seeweapon",
    "shell",
    "sit",
    "swap",
    "takeoff",
    "takeoffall",
    "teleport",
    "throw",
    "tip",
    "travel",
    "turnundead",
    "twoweapon",
    "untrap",
    "version",
    "versionshort",
    "wear",
    "whatdoes",
    "whatis",
    "wield",
    "wipe",
    "zap",
    "minus",
    "space",
    "apos",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "wizard detect",
    "wizard genesis",
    "wizard identify",
    "wizard teleport",
    "wizard map",
    "wizard wish",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
    "far north",
    "far east",
    "far south",
    "far west",
    "far northeast",
    "far southeast",
    "far southwest",
    "far northwest",
    "+",
    '"',
    "$",
]
