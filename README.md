A template repository which can be used to test aggregation algorithms. I provide reference implementations for loading worker estimates and baseline policies like Averaging and supervised learning. I will add more details later.

### Create python env
1. On Windows, run `conda env create -f env_windows.yml`
2. On Linux, run `conda env create -f env_linux.yml`

These will create a env with name `editable_proj`. To change the name, edit the `name` field in the yml file. To activate the env, run `conda activate editable_proj`

### Install the package
1. Run `pip install -e .` in the root directory of the project. This will install the package in editable mode.

### Run the tests
1. Run `pytest` in the root directory of the project. This will run all the tests in the `tests` directory.

### Running code without wandb
python main.py main.use_wandb=False policy=averaging policy.params.reg=0.1
python main.py policy=sl_agg main.use_wandb=False mlp.params.hidden_size=10