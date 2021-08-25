## Generate training data

Training data can be generated using the `generate_FKset.py` python script.

1. Clone this repo with `git clone https://github.com/epignatelli/cardiax.git`
1. Install jax with `sh cardiax/install_jax.sh`, or follow the instructions at https://github.com/google/jax#installation. The `install_jax.sh` file is at https://github.com/epignatelli/cardiax/blob/master/install_jax.sh
1. Install requirements with `pip install -r cardiax/requirements.txt`. The requirements file is at https://github.com/epignatelli/cardiax/blob/master/requirements.txt
1. Install the `cardiax` package with `pip install -e cardiax`
1. Generate the training data with `python cardiax/experiments/training/generate_FKset.py`

