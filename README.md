# CleanQRL (Clean Quantum Reinforcement Learning)


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/fhg-iisb/cleanqrl?tab=License-1-ov-file)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)](https://fhg-iisb.github.io/cleanqrl-docs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**CleanQRL** is a Reinforcement Learning library specifically tailored to the subbranch of Quantum Reinforcement Learning and is greatly inspired by the amazing work of **[CleanRL](https://github.com/vwxyzjn/cleanrl)**. Just as the classical analogue, we aim to provide high-quality single-file implementation with research-friendly features. The implementation follows mainly the ideas of **[CleanRL](https://github.com/vwxyzjn/cleanrl)** and is clean and simple, yet can scale nicely trough additional features such as **[ray tune](https://docs.ray.io/en/latest/tune/index.html)**. The main features of this repository are


* 📜 Single-file implementations of classical and quantum version of 4+ Reinforcement Learning agents 
* 💾 Tuned and Benchmarked agents (with available configs)
* 🎮 Integration of [gymnasium](https://gymnasium.farama.org/), [mujoco](https://www.gymlibrary.dev/environments/mujoco/index.html) and [jumanji](https://instadeepai.github.io/jumanji/)
* 📘 Examples on how to enhance the standard QRL agents on a variety of games
* 📈 Tensorboard Logging
* 🌱 Local Reproducibility via Seeding
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 📊 Easy and straight forward hyperparameter tuning with [ray tune](https://docs.ray.io/en/latest/tune/index.html)

What we are missing compared to **[CleanRL](https://github.com/vwxyzjn/cleanrl)**:

* 💸 Cloud Integration with docker and AWS 
* 📹 Videos of Gameplay Capturing


You can read more about **CleanQRL** in [our upcoming paper](https://arxiv.org/pdf/2507.07593).

# Get started

## Installation

To run experiments locally, you need to clone the repository and install a python environment.

```bash
git clone https://github.com/fhg-iisb/cleanqrl.git
cd cleanqrl
conda env create -f environment.yaml
```

Thats it, now you're set up!

## Run first experiments

Each agent can be run as a single file, either from the parent directory or directly in the subfolder. First, activate the environment ```cleanqrl``` and then execute the algorithm's python file:

```
conda activate cleanqrl
python cleanqrl/reinforce_quantum.py 
```

or go directly into the folder and execute

```
conda activate cleanqrl
cd cleanqrl 
python reinforce_quantum.py 
```

Before you execute the files, customize the parameters in the  ```Config``` class at the end of each file. Every file has such a dataclass object and the algorithm is callable as a function which takes the config as input:


```python
def reinforce_quantum(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]
    .... 
```

This function can also be called from an external file (see below for details). But first, lets have a closer look to the the ```Config```: 

```py title="reinforce_quantum.py"
@dataclass
class Config:
    # General parameters
    trial_name: str = 'reinforce_quantum'  # Name of the trial
    trial_path: str = 'logs'  # Path to save logs relative to the parent directory
    wandb: bool = False # Use wandb to log experiment data 
    project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

    # Environment parameters
    env_id: str = "CartPole-v1" # Environment ID
    
    # Algorithm parameters
    num_envs: int = 1  # Number of environments
    total_timesteps: int = 100000  # Total number of timesteps
    gamma: float = 0.99  # discount factor
    lr_input_scaling: float = 0.01  # Learning rate for input scaling
    lr_weights: float = 0.01  # Learning rate for variational parameters
    lr_output_scaling: float = 0.01  # Learning rate for output scaling
    cuda: bool = False  # Whether to use CUDA
    num_qubits: int = 4  # Number of qubits
    num_layers: int = 2  # Number of layers in the quantum circuit
    device: str = "default.qubit"  # Quantum device
    diff_method: str = "backprop"  # Differentiation method
    save_model: bool = True # Save the model after the run

```

As you can see, the config is divided into 3 parts:

* **General parameters**: Here the name of your experiment as well as the logging path is defined. All metrics will be logged in a ```result.json``` file in the result folder which will have the time of the experiment execution as a prefix. You can also use [wandb](https://wandb.ai/site) for enhanced metric logging. 
* **Environment parameters**: This is in the simplest case just the string of the gym environment. For jumanji environments as well as for your custom environments, you can also specify additional parameters here (see #Tutorials for details).
* **Algorithms parameters**: All algorithms hyperparameters are specified here. For details on the parameters see [the algorithms section](https://fhg-iisb.github.io/cleanqrl-docs/algorithms/overview/)

Once you execute the file, it will create the subfolders and copy the config which is used for the experiment in the folder:

```py title="reinforce_quantum.py"
    config = vars(Config())
    
    # Based on the current time, create a unique name for the experiment
    config['trial_name'] = datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config['trial_name']
    config['path'] = os.path.join(Path(__file__).parent.parent, config['trial_path'], config['trial_name'])

    # Create the directory and save a copy of the config file so that the experiment can be replicated
    os.makedirs(os.path.dirname(config['path'] + '/'), exist_ok=True)
    config_path = os.path.join(config['path'], 'config.yml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # Start the agent training 
    reinforce_quantum(config)   
```

After the execution, the experiment data is saved e.g. at: 

    ...
    configs
    examples
    logs/
        2025-03-04--14-59-32_reinforce_quantum          # The name of your experiment
            config.yaml                                 # Config which was used to run this experiment
            result.json                                 # Results of the experiment
    .gitignore
    ...


You can also set the ```wandb``` variable to ```True```:

```py title="reinforce_quantum.py" hl_lines="4 6"
@dataclass
class Config:
    # General parameters
    trial_name: str = 'reinforce_quantum_wandb'  # Name of the trial
    trial_path: str = 'logs'  # Path to save logs relative to the parent directory
    wandb: bool = True # Use wandb to log experiment data 
    project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

    # Environment parameters
    env_id: str = "CartPole-v1" # Environment ID
```

You will need to login to your [wandb](https://wandb.ai/site) account before you can run:

```bash
wandb login # only required for the first time
python cleanqrl/reinforce_quantum.py \
```

This will create an additional folder for the [wandb](https://wandb.ai/site) logging and you can inspect your experiment data also online:

    ...
    configs
    examples
    logs/
        2025-03-04--14-59-32_reinforce_quantum_wandb    # The name of your experiment
            wandb                                       # Subfolder of the wandb logging
            config.yaml                                 # Config which was used to run this experiment
            result.json                                 # Results of the experiment
    .gitignore
    ...

# Algorithms

## Contact and Community

We want to grow as a community, so posting [Github Issues](https://github.com/fhg-iisb-mki/cleanqrl/issues) and PRs are very welcome! If you are missing and algorithms or have a specific problem to which you want to tailor your QRL algorithms but fail to do so, you can also create a feature request!

## Citing CleanQRL

If you use **CleanQRL** in your work, please cite our [paper](https://fhg-iisb.github.io/cleanqrl-docs/algorithms/overview/):

```bibtex
@article{kruse2025cleanqrl,
  title={CleanQRL: Lightweight Single-file Implementations of Quantum Reinforcement Learning Algorithms},
  author={Kruse, Georg and Coelho, Rodrigo and Rosskopf, Andreas and Wille, Robert and Lorenz, Jeanette Miriam},
  journal={arXiv preprint arXiv:2507.07593},
  year={2025}
}
```

## Citing CleanRL

If you used mainly the classical parts of our code in your work, please cite the original [CleanRL paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```