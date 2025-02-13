A template repository which can be used to test aggregation algorithms. I provide reference implementations for loading worker estimates and baseline policies like Averaging and supervised learning. I will add more details later.

### Create python env and package with name `my_proj`
For this, we will change the name of the package and the environment.
1. Change the name of the package in `pyproject.toml` and `editable_proj` directory to `my_proj`.
2. Change the name of the environment in `env_windows.yml` and `env_linux.yml` to `my_proj`.
3. Change the name of the package in the `main.py` file and files in `tests\` to `my_proj`.

Now, to create a conda environment and install the package, follow the steps below.
1. Conda env creation. 
- On Windows, run `conda env create -f env_windows.yml`
- On Linux, run `conda env create -f env_linux.yml`
2. Run `conda activate my_proj` to activate the environment.
3. Run `pip install -e .` in the root directory of the project. This will install the package in editable mode.

### Run the tests
1. Run `pytest` in the root directory of the project. This will run all the tests in the `tests` directory.

### Running code without wandb
python main.py main.use_wandb=False policy=averaging policy.params.reg=0.1
python main.py policy=sl_agg main.use_wandb=False mlp.params.hidden_size=10

### WandB
To use wandb, you need to change the following.
1. In `conf/config.yaml`, change `wandb.entity` from `Null` to your wandb entity.
2. In `conf/config.yaml`, change `wandb.project` from `Null` to your wandb project.
3. Pass `main.use_wandb=True` in the command line arguments.