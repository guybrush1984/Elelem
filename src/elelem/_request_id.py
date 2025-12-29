"""
Human-readable request ID generation.

Generates IDs in format: word-1234 (e.g., "blue-4829", "frog-1573")
"""

import random

# ~500 common, easy-to-read 4-letter words
WORDS = [
    # Animals
    "bear", "bird", "buck", "bull", "calf", "clam", "crab", "crow", "deer", "dove",
    "duck", "fawn", "fish", "flea", "frog", "goat", "gull", "hare", "hawk", "lion",
    "lynx", "mink", "mole", "moth", "mule", "newt", "puma", "seal", "slug", "swan",
    "toad", "wasp", "wolf", "worm", "wren",

    # Colors
    "aqua", "blue", "cyan", "gold", "gray", "grey", "jade", "lime", "navy", "onyx",
    "pink", "plum", "rose", "ruby", "rust", "sage", "sand", "teal", "wine",

    # Nature
    "bark", "bead", "bolt", "cave", "clay", "coal", "cove", "dune", "dust", "fern",
    "fire", "foam", "frost", "glen", "gust", "hail", "hill", "isle", "lake", "lava",
    "leaf", "mesa", "mist", "moon", "moss", "oaks", "palm", "peak", "pine", "pond",
    "rain", "reed", "rift", "rock", "root", "sand", "seed", "snow", "star", "stem",
    "surf", "tide", "tree", "vale", "vine", "wave", "weed", "wind", "wood",

    # Time/Weather
    "calm", "dawn", "dusk", "fall", "gale", "haze", "noon", "rise", "warm", "year",

    # Objects
    "arch", "axle", "ball", "band", "bank", "barn", "beam", "bell", "belt", "boat",
    "bolt", "book", "bowl", "bulb", "cage", "cake", "card", "cart", "case", "chip",
    "clip", "club", "coat", "coil", "coin", "comb", "cone", "cord", "cork", "crib",
    "cube", "cups", "curl", "dart", "desk", "dial", "dice", "disc", "dish", "dock",
    "dome", "door", "drum", "edge", "flag", "fork", "fuse", "gate", "gear", "gift",
    "grid", "grip", "grit", "halo", "harp", "helm", "hook", "horn", "hose", "iron",
    "jack", "jars", "keys", "kite", "knob", "knot", "lace", "lamp", "lane", "lens",
    "lift", "link", "lock", "logo", "loop", "mast", "maze", "mesh", "mill", "mint",
    "nail", "nest", "node", "note", "oven", "pack", "pail", "pane", "path", "pawn",
    "pins", "pipe", "pole", "pool", "post", "pump", "rack", "rail", "ramp", "ring",
    "road", "robe", "rope", "rug", "sack", "sail", "seal", "seat", "shed", "ship",
    "shoe", "shop", "sign", "sink", "slab", "slot", "soap", "sock", "sofa", "spin",
    "step", "stud", "suit", "tank", "tape", "tent", "tile", "tint", "tire", "tray",
    "tube", "vase", "vent", "vest", "vial", "wall", "wick", "wing", "wire", "wrap",

    # Food/Plants
    "bean", "beet", "brew", "cake", "chip", "chop", "corn", "date", "drop", "figs",
    "herb", "kale", "kiwi", "leek", "lime", "meal", "meat", "milk", "nuts", "oats",
    "pear", "peas", "plum", "rice", "roll", "salt", "soup", "tart", "yams", "zest",

    # Actions/Verbs (as nouns)
    "bang", "bash", "beam", "beat", "bend", "bite", "blur", "boom", "bump", "buzz",
    "call", "chat", "chop", "clap", "copy", "dash", "deal", "dent", "dive", "drag",
    "drip", "drop", "echo", "fade", "fall", "find", "fizz", "flap", "flip", "flow",
    "fold", "fuel", "gain", "gaze", "glow", "grab", "grow", "gulp", "hack", "halt",
    "hang", "haul", "heap", "help", "hide", "hint", "hold", "honk", "hook", "hoop",
    "hope", "howl", "huff", "hunt", "hurl", "jolt", "jump", "kick", "knit", "lead",
    "leak", "lean", "leap", "lift", "limp", "link", "list", "load", "look", "loom",
    "loop", "loot", "lull", "lump", "lure", "mark", "mash", "meet", "melt", "mend",
    "move", "nods", "note", "pace", "pack", "pass", "pats", "peck", "peek", "peel",
    "pick", "ping", "plan", "play", "plop", "plot", "plow", "plug", "plum", "poke",
    "poll", "pond", "poof", "pool", "pour", "prod", "prop", "puff", "pull", "pump",
    "push", "race", "raid", "rant", "rest", "ride", "riff", "rift", "rise", "risk",
    "roam", "roar", "rock", "roll", "rush", "rust", "sail", "scan", "seal", "seep",
    "send", "shed", "ship", "shop", "shot", "show", "shut", "sigh", "sign", "sink",
    "sips", "skip", "slam", "slap", "slip", "snap", "snip", "soar", "sort", "span",
    "spin", "spit", "spot", "stab", "stay", "stem", "step", "stew", "stir", "stop",
    "stow", "stun", "sway", "swim", "swip", "tack", "taps", "task", "tear", "test",
    "tick", "tilt", "tint", "toss", "tour", "trap", "trek", "trim", "trip", "trot",
    "tuck", "tug", "turn", "veer", "vent", "view", "volt", "vote", "wade", "waft",
    "wail", "wait", "wake", "walk", "wane", "warm", "warn", "warp", "wash", "wave",
    "weep", "weld", "whip", "wilt", "wink", "wipe", "wish", "wrap", "yank", "yawn",
    "yelp", "yell", "yoga", "zero", "zest", "zing", "zoom",

    # Adjectives (as identifiers)
    "bold", "boxy", "busy", "calm", "cozy", "crisp", "cute", "damp", "dark", "deep",
    "easy", "epic", "even", "fair", "fast", "firm", "flat", "free", "full", "glad",
    "gold", "good", "half", "hard", "hazy", "high", "huge", "idle", "keen", "kind",
    "lazy", "lean", "lite", "live", "long", "loud", "lush", "main", "mega", "mild",
    "mint", "neon", "neat", "nice", "open", "pale", "pure", "quad", "rare", "real",
    "rich", "ripe", "safe", "slim", "slow", "snug", "soft", "solo", "tall", "tame",
    "thin", "tiny", "true", "twin", "vast", "warm", "wide", "wild", "wise", "zero",
]

# Deduplicate (some words appear in multiple categories)
WORDS = list(dict.fromkeys(WORDS))


def generate_request_id() -> str:
    """Generate a human-readable request ID.

    Format: word-NNNN (e.g., "blue-4829", "frog-1573")

    With ~450 words and 10000 number combinations, we get ~4.5M unique IDs.
    Collision probability is low for typical request volumes.
    """
    word = random.choice(WORDS)
    number = random.randint(0, 9999)
    return f"{word}-{number:04d}"
