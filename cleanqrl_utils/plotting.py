import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_single_run(path):
    # Load the results.json file
    results_path = os.path.join(path, "result.json")

    results = pd.read_json(results_path, lines=True)

    results_selected_returns = results[pd.notna(results["episodic_return"])]
    episode_returns = results_selected_returns["episodic_return"].values
    global_steps_rewards = results_selected_returns["global_step"].values

    results_selected_loss = results[pd.notna(results["loss"])]
    episode_losses = results_selected_loss["loss"].values
    global_steps_loss = results_selected_loss["global_step"].values

    # Plot episode_return vs. global_step
    plt.figure(figsize=(10, 5))
    plt.plot(global_steps_rewards, episode_returns, label="Episode Return")
    plt.xlabel("Global Step")
    plt.ylabel("Episode Return")
    plt.title("Episode Return vs. Global Step")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "training_results_reward.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(global_steps_loss, episode_losses, label="Episode Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Episode Loss")
    plt.title("Episode Return vs. Global Step")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "training_results_loss.png"))


def plot_tune_run(path):
    fig, axis = plt.subplots(figsize=(8, 7))

    paths, labels, types = [], [], []
    paths += [path]

    title = "4 qubits local grid 1 env"
    fig_name = f"{title}"
    data_length = 300

    labels += [["0.1", "0.05", "0.01", "0.005", "0.001"]]

    types += [
        {
            "0": ["=0.1"],
            "1": ["=0.05"],
            "2": ["=0.01"],
            "3": ["=0.005"],
            "4": ["=0.001"],
        }
    ]

    for idx, path in enumerate(paths):
        # path = os.path.basename(os.path.normpath(path))
        curves = [[] for _ in range(len(labels[idx]))]
        curves_x = [[] for _ in range(len(labels[idx]))]

        results_file_name = "/result.json"
        result_files = [f.path for f in os.scandir(path) if f.is_dir()]
        results = [
            pd.read_json(f + results_file_name, lines=True) for f in result_files
        ]
        result = pd.concat(results)

        for key, value in types[idx].items():
            for i, data_exp in enumerate(results):
                if all(x in result_files[i] for x in value):
                    curves_x[int(key)].append(data_exp["episode"].values[:data_length])
                    curves[int(key)].append(
                        data_exp["episodic_return"].values[:data_length]
                    )

        for id, curve in enumerate(curves):
            data = np.vstack(curve)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            upper = mean + std
            lower = mean - std
            axis.plot(
                data_exp["episode"].values[:data_length], mean, label=labels[idx][id]
            )
            axis.fill_between(
                data_exp["episode"].values[:data_length], lower, upper, alpha=0.5
            )

    axis.set_xlabel("$Episodes$", fontsize=13)
    axis.set_ylabel("$Return$", fontsize=15)
    axis.set_title(title, fontsize=15)
    axis.legend(fontsize=12, loc="lower left")
    axis.minorticks_on()
    axis.grid(which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=100)


if __name__ == "__main__":
    path = "C:\\Users\\kruse\\code\\standardrl\\logs\\2025-02-04--15-01-56_RL_"
    # path = 'H:\\standardrl\\logs\\2025-02-03--14-18-00_QRL_\\train_agent_2025-02-03_14-18-00\\'
    plot_single_run(path)
    # plot_tune_run(path)
