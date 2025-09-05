# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from replay_buffer import ReplayBuffer, ReplayBufferWrapper
from wrapper_jumanji import create_jumanji_env


# ENV LOGIC: create your env (with config) here:
def make_env(env_id: str, config: dict):
    def thunk():
        env = create_jumanji_env(env_id, config)
        env = ReplayBufferWrapper(env)
        return env

    return thunk


# ALGO LOGIC: initialize your agent here:
class DQNAgentClassical(nn.Module):
    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def log_metrics(config, metrics, report_path=None):
    if config["wandb"]:
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        with open(os.path.join(report_path, "result.json"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


# MAIN TRAINING FUNCTION
def dqn_classical_jumanji(config: dict):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    buffer_size = config["buffer_size"]
    total_timesteps = config["total_timesteps"]
    start_e = config["start_e"]
    end_e = config["end_e"]
    exploration_fraction = config["exploration_fraction"]
    learning_starts = config["learning_starts"]
    train_frequency = config["train_frequency"]
    batch_size = config["batch_size"]
    gamma = config["gamma"]
    target_network_frequency = config["target_network_frequency"]
    tau = config["tau"]
    lr = config["lr"]

    if config["seed"] == "None":
        config["seed"] = None

    if not ray.is_initialized():
        report_path = config["path"]
        name = config["trial_name"]
        with open(os.path.join(report_path, "result.json"), "w") as f:
            f.write("")
    else:
        session = get_session()
        report_path = session.storage.trial_fs_path
        name = session.storage.trial_fs_path.split("/")[-1]

    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            sync_tensorboard=True,
            config=config,
            name=name,
            monitor_gym=True,
            save_code=True,
            dir=report_path,
        )
    # TRY NOT TO MODIFY: seeding
    if config["seed"] is None:
        seed = np.random.randint(0, 1e9)
    else:
        seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(env_id, config) for i in range(num_envs)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    observation_size = np.array(envs.single_observation_space.shape).prod()
    num_actions = envs.single_action_space.n

    # Here, the classical agent is initialized with a Neural Network
    q_network = DQNAgentClassical(observation_size, num_actions).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    target_network = DQNAgentClassical(observation_size, num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # global parameters to log
    print_interval = 50
    global_episodes = 0
    episode_returns = deque(maxlen=print_interval)
    episode_approximation_ratio = deque(maxlen=print_interval)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            start_e, end_e, exploration_fraction * total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            # action masking: 
            # action_mask = torch.tensor(obs[:, :envs.single_action_space.n], dtype=torch.float32)
            # large_negative = -1e2  # Large negative value for invalid actions
            # q_values_masked = q_values + (1 - action_mask) * large_negative            
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    metrics = {}
                    global_episodes += 1
                    episode_returns.append(infos["episode"]["r"].tolist()[idx])
                    metrics["episode_reward"] = infos["episode"]["r"].tolist()[idx]
                    metrics["episode_length"] = infos["episode"]["l"].tolist()[idx]
                    metrics["global_step"] = global_step
                    if "approximation_ratio" in infos.keys():
                        if infos["approximation_ratio"][idx]:
                            metrics["approximation_ratio"] = infos["approximation_ratio"][
                                idx
                            ]
                            episode_approximation_ratio.append(
                                metrics["approximation_ratio"]
                            )
                    log_metrics(config, metrics, report_path)

            if global_episodes % print_interval == 0 and not ray.is_initialized():
                logging_info = f"Global step: {global_step}  Mean return: {np.mean(episode_returns)}"
                if len(episode_approximation_ratio) > 0:
                    logging_info += f"  Mean approximation ratio: {np.mean(episode_approximation_ratio)}"
                print(logging_info)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                data = rb.sample(batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.float()).max(
                        dim=1
                    )
                    td_target = data.rewards.flatten() + gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = (
                    q_network(data.observations.float())
                    .gather(1, data.actions)
                    .squeeze()
                )
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    metrics = {}
                    metrics["td_loss"] = loss.item()
                    metrics["q_values"] = old_val.mean().item()
                    metrics["SPS"] = int(global_step / (time.time() - start_time))
                    log_metrics(config, metrics, report_path)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        tau * q_network_param.data
                        + (1.0 - tau) * target_network_param.data
                    )

    if config["save_model"]:
        model_path = f"{os.path.join(report_path, name)}.cleanqrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if config["wandb"]:
        wandb.finish()


if __name__ == "__main__":

    @dataclass
    class Config:
        # General parameters
        trial_name: str = "dqn_classical_jumanji"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "TSP-v1"  # Environment ID 
        num_cities: int = 4

        # Algorithm parameters
        num_envs: int = 1  # Number of environments
        seed: int = None  # Seed for reproducibility
        buffer_size: int = 10000  # Size of the replay buffer
        total_timesteps: int = 100000  # Total number of timesteps
        start_e: float = 0.01  # Starting value of epsilon for exploration
        end_e: float = 0.01  # Ending value of epsilon for exploration
        exploration_fraction: float = 0.1  # Fraction of total timesteps for exploration
        learning_starts: int = 1000  # Timesteps before learning starts
        train_frequency: int = 1  # Frequency of training
        batch_size: int = 32  # Batch size for training
        gamma: float = 0.99  # Discount factor
        target_network_frequency: int = 100  # Frequency of target network updates
        tau: float = 0.01  # Soft update coefficient
        lr: float = 0.01  # Learning rate for network weights
        cuda: bool = False  # Whether to use CUDA
        save_model: bool = True  # Save the model after the run

    config = vars(Config())

    # Based on the current time, create a unique name for the experiment
    config["trial_name"] = (
        datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + "_" + config["trial_name"]
    )
    config["path"] = os.path.join(
        Path(__file__).parent.parent, config["trial_path"], config["trial_name"]
    )
    # Create the directory and save a copy of the config file so that the experiment can be replicated
    os.makedirs(os.path.dirname(config["path"] + "/"), exist_ok=True)
    config_path = os.path.join(config["path"], "config.yml")
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Start the agent training
    dqn_classical_jumanji(config)
