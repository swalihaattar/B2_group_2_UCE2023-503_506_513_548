# environment/levels.py

# Each level has 2 variations
# Format: [variation1, variation2]

BEGINNER_LEVELS = [
    # Variation 1 - Simple open maze
    [
        "%%%%%%%%%",
        "%.P    .%",
        "% %..G%%%",  
        "%.o   . %",
        "%%%%%%%%%",
    ],
    # Variation 2 - Similar complexity, different layout
    [
        "%%%%%%%%%%%%%",
        "%P .   %   .%",
        "% %%% % %%% %",
        "% .   %     %",
        "% % %%% %%% %",
        "%o  .   .  G%",
        "%%%%%%%%%%%%%"
    ]
]

INTERMEDIATE_LEVELS = [
    # Variation 1 - More walls, tighter spaces
    [
        "%%%%%%%%%%%%%%%%%%%",
        "%..       G   ....%",
        "%.%.  %%%%%% %.%%.%",
        "%.%% o%   o. %.o%.%",
        "%.%%%.%  %%% %..%.%",
        "%.....  P    ....G%",
        "%%%%%%%%%%%%%%%%%%%",
    ],
    # Variation 2 - Different intermediate layout
    [
        "%%%%%%%%%%%%%%%",
        "%.....G G.....%",
        "%.%.%o   o%.%.%",
        "%...%.....%...%",
        "%.%...% %...%.%",
        "%......P......%",
        "%%%%%%%%%%%%%%%"
    ]
]

PRO_LEVELS = [
    # Variation 1 - Complex maze, many corridors
    [
        "%%%%%%%%%%%%%%%%%%%%",
        "%oG..%........%..Go%",
        "%.%.....%..%....%%.%",
        "%.%.....%..%.....%.%",
        "%.%.%%.%%  %%.%%.%.%",
        "%...... .G  %.%....%",
        "%.%....%  ..%.%..%.%",
        "%.%....%    %.%..%.%",
        "%.%...........%....%",
        "%...%.%%%%%%...%%..%",
        "%o...%...P........o%",
        "%%%%%%%%%%%%%%%%%%%%"
    ],
    # Variation 2 - Different pro layout
    [
        "%%%%%%%%%%%%%%%%%%%%",
        "%....%.o......%....%",
        "%.%%.%.%%%%%%.%.%%.%",
        "%.%..............%.%",
        "%.%.%%..%  %.o%%.%.%",
        "%......%G GG%......%",
        "%.%.o...%%%%..%%.%.%",
        "%.%..............%.%",
        "%.%%.%.%%%%%%.%.%%.%",
        "%....%...P....%...o%",
        "%%%%%%%%%%%%%%%%%%%%"
    ]
]

# Map level names to their variations
LEVELS = {
    "beginner": BEGINNER_LEVELS,
    "intermediate": INTERMEDIATE_LEVELS,
    "pro": PRO_LEVELS
}

# Level progression order
LEVEL_ORDER = ["beginner", "intermediate", "pro"]

# Max points per level (for balancing)
LEVEL_MAX_POINTS = {
    "beginner": 200,      # Adjust based on your maze pellet count
    "intermediate": 350,  # Adjust based on your maze pellet count
    "pro": 500           # Adjust based on your maze pellet count
}