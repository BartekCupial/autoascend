from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "num_assessments": 128,
    "num_threads": 16,
    "game": "NetHackChallenge-v0",
    "savedir": "data/nle_data",
    "gamesavedir": "data/demos",
    "seed": False,
    "save_video": False,
    "save_demo": True,
}

# params different between exps
params_grid = [
    {
        "experiment_id": list(range(5)),
    }
]


experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="autoascend",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
