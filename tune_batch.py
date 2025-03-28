import os
import shutil
from datetime import datetime

import ray
import yaml
from ray import tune

from cleanqrl_utils.config import add_hyperparameters
from cleanqrl_utils.plotting import plot_tune_run
from cleanqrl_utils.train import train_agent

if __name__ == "__main__":
    # Specify the path to the config file
    config_paths = ["configs/ppo_classical.yaml"]

    for config_path in config_paths:
        # Load the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Generate the parameter space for the experiment from the config file
        config = add_hyperparameters(config)

        # Based on the current time, create a unique name for the experiment
        config["trial_name"] = (
            datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + "_" + config["trial_name"]
        )
        config["path"] = os.path.join(
            os.getcwd(), config["trial_path"], config["trial_name"]
        )

        # Create the directory and save a copy of the config file so
        # that the experiment can be replicated
        os.makedirs(os.path.dirname(config["path"] + "/"), exist_ok=True)
        shutil.copy(config_path, config["path"] + "/config.yml")

        # Instead of running a single agent as before, we will use ray.tune to run multiple agents
        # in parallel. We will use the same train_agent function as before.
        ray.init(
            local_mode=config["ray_local_mode"],
            num_cpus=config["num_cpus"],
            num_gpus=config["num_gpus"],
            _temp_dir=os.path.join(os.path.dirname(os.getcwd()), "t"),
            include_dashboard=False,
        )

        # We need an addtional function to create subfolders for each hyperparameter configuration
        def trial_name_creator(trial):
            return trial.__str__() + "_" + trial.experiment_tag

        # We will use the tune.Tuner class to run multiple agents in parallel
        trainable = tune.with_resources(
            train_agent,
            resources={
                "cpu": config["cpus_per_worker"],
                "gpu": config["gpus_per_worker"],
            },
        )
        tuner = tune.Tuner(
            trainable,
            param_space=config,
            run_config=tune.RunConfig(storage_path=config["path"]),
            tune_config=tune.TuneConfig(
                num_samples=config["num_samples"],
                trial_dirname_creator=trial_name_creator,
            ),
        )

        # The fit function will start the hyperparameter search
        tiral = tuner.fit()

        # After the experiment is done, we will plot the results.
        ray.shutdown()
        # plot_tune_run(path)
