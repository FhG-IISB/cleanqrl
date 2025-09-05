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
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
# ENV LOGIC: create your env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=config["gamma"])
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


# ALGO LOGIC: initialize your agent here:
class EvolutionAgent(nn.Module):
    def __init__(self, observation_size, num_actions, discretize=False, num_bins=0, bin_centers=None):
        super().__init__()
        self.discretize = discretize
        if self.discretize:
            self.num_bins = num_bins
            self.bin_centers = torch.tensor(bin_centers)
            out_size = num_actions * num_bins
        else:
            out_size = num_actions
        self.actor = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
        )
        if not self.discretize:
            self.actor.add_module("tanh", nn.Tanh())

    def get_action(self, x):
        output = self.actor(x)
        if self.discretize:
            logits = output.view(x.shape[0], -1, self.num_bins)
            indices = torch.argmax(logits, dim=-1)
            actions = torch.gather(self.bin_centers.unsqueeze(0).expand(x.shape[0], -1, -1), 2, indices.unsqueeze(2)).squeeze(2)
        else:
            actions = output
        return actions
    
def get_flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        sz = p.numel()
        p.data.copy_(flat_params[idx:idx + sz].view_as(p))
        idx += sz


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
def es_classical_continuous_action(config):
    num_directions = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    alpha = config["lr"]
    gamma = config["gamma"]
    sigma = config["sigma"]
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
    if config["seed"] is None:
        seed = np.random.randint(0, 1e9)
    else:
        seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config["cuda"]) else "cpu"
    )
    assert (
        env_id in gym.envs.registry.keys()
    ), f"{env_id} is not a valid gymnasium environment"
    temp_env = make_env(env_id, config)()
    assert isinstance(
        temp_env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    assert isinstance(
        temp_env.observation_space, gym.spaces.Box
    ), "only continuous observation space is supported"
    observation_size = np.array(temp_env.observation_space.shape).prod()
    num_actions = np.prod(temp_env.action_space.shape)
    low = temp_env.action_space.low
    high = temp_env.action_space.high
    temp_env.close()
    discretize = env_id in ["Hopper-v5", "Swimmer-v5"]
    num_bins = 10 if discretize else 0
    if discretize:
        bin_centers = []
        for d in range(num_actions):
            edges = np.linspace(low[d], high[d], num_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            bin_centers.append(centers)
        bin_centers = np.array(bin_centers)
    else:
        bin_centers = None
    # Here, the agent is initialized with a Neural Network
    agent = EvolutionAgent(observation_size, num_actions, discretize, num_bins, bin_centers).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=alpha)
    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 50
    episode_returns = deque(maxlen=print_interval)
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    num_evals = 2 * num_directions
    env_fns = [make_env(env_id, config) for _ in range(num_evals)]
    envs = [fn() for fn in env_fns]
    while global_step < total_timesteps:
        metrics = {}
        theta = get_flat_params(agent)
        param_size = theta.size(0)
        epsilons = torch.randn(num_directions, param_size, device=device)
        pert_list = sigma * epsilons
        pert_list_neg = -sigma * epsilons
        pert_list = torch.cat((pert_list, pert_list_neg), dim=0)
        agents = []
        for i in range(num_evals):
            perturbed_agent = EvolutionAgent(observation_size, num_actions, discretize, num_bins, bin_centers).to(device)
            set_flat_params(perturbed_agent, theta + pert_list[i])
            agents.append(perturbed_agent)
        # Rollout each perturbed policy
        fitness = []
        ep_lengths = []
        for i in range(num_evals):
            env = envs[i]
            perturbed_agent = agents[i]
            obs, _ = env.reset()
            obs = torch.Tensor(obs).to(device)
            done = False
            while not done:
                action = perturbed_agent.get_action(obs.unsqueeze(0)).squeeze(0).cpu().numpy()
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                obs = torch.Tensor(next_obs).to(device)
                if done and "episode" in info:
                    fit = info["episode"]["r"]
                    length = info["episode"]["l"]
            fitness.append(fit)
            ep_lengths.append(length)
            global_step += length
            global_episodes += 1
        fitness = np.array(fitness)
        # Compute utilities with fitness shaping
        m = num_evals
        ranked_indices = np.argsort(-fitness)  # descending
        ranks = np.empty(m)
        ranks[ranked_indices] = np.arange(m)
        sum_util = 0.0
        for j in range(1, m + 1):
            sum_util += max(0, np.log(m / 2 + 1) - np.log(j))
        utilities = np.array([max(0, np.log(m / 2 + 1) - np.log(r + 1)) / (sum_util + 1e-9) - 1 / m for r in ranks])
        u_pos = utilities[:num_directions]
        u_neg = utilities[num_directions:]
        # Compute gradient estimate
        diff_u = torch.tensor(u_pos - u_neg, dtype=torch.float32, device=device).unsqueeze(1)
        sum_eps = (diff_u * epsilons).sum(dim=0)
        g = sum_eps / (2 * num_directions * sigma)
        # Update the policy
        optimizer.zero_grad()
        idx = 0
        for p in agent.parameters():
            sz = p.numel()
            p.grad = -g[idx:idx + sz].view_as(p).clone()  # negative for maximization
            idx += sz
        optimizer.step()
        # Logging
        mean_return = np.mean(fitness)
        episode_returns.append(mean_return)
        metrics["episode_reward"] = mean_return
        metrics["episode_length"] = np.mean(ep_lengths)
        metrics["global_step"] = global_step
        metrics["SPS"] = int(global_step / (time.time() - start_time))
        log_metrics(config, metrics, report_path)
        if len(episode_returns) % print_interval == 0 and not ray.is_initialized():
            print(
                "Global step: ",
                global_step,
                " Mean return: ",
                np.mean(episode_returns),
            )
    if config["save_model"]:
        model_path = f"{os.path.join(report_path, name)}.cleanqrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    for env in envs:
        env.close()
    if config["wandb"]:
        wandb.finish()
if __name__ == "__main__":
    @dataclass
    class Config:
        # General parameters
        trial_name: str = "es_classical_continuous_action"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project
        # Environment parameters
        env_id: str = "Hopper-v5"  # Environment ID
        # Algorithm parameters
        num_envs: int = 1  # Number of directions (n)
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 200000  # Total number of timesteps
        gamma: float = 0.9  # discount factor
        lr: float = 0.001  # Learning rate (alpha) for Adam
        sigma: float = 0.01  # Noise standard deviation
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
    es_classical_continuous_action(config)