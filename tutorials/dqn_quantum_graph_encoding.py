# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import datetime
import itertools
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
import pennylane as qml
import ray
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from jumanji import wrappers
from jumanji.environments import TSP
from jumanji.environments.routing.tsp.generator import UniformGenerator
from ray.train._internal.session import get_session
from replay_buffer import ReplayBuffer, ReplayBufferWrapper


# We need to create a new wrapper for the Knapsack environment that converts
# the observation into a cost hamiltonian of the problem.
class JumanjiWrapperTSP(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # For the knapsack problem we use the so called unbalanced penalization method
        # This means that we will have sum(range(num_items)) quadratic terms + num_items linear terms
        # This is constant throughout
        self.num_cities = self.env.unwrapped.num_cities
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(sum(range(self.num_cities)) + self.num_cities,)
        )
        self.episodes = 0

    def reset(self, **kwargs):
        state, info = self.env.reset()
        # convert the state to cost hamiltonian
        pairwise_distances = self.calculate_city_distances(state["coordinates"])
        annotations = state["trajectory"]
        state = np.hstack([pairwise_distances, annotations])
        return state, info

    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate:
            if self.episodes % 100 == 0:
                nodes = state["coordinates"]
                optimal_tour_length = self.tsp_compute_optimal_tour(nodes)
                info["optimal_tour_length"] = optimal_tour_length
                info["approximation_ratio"] = info["episode"]["r"] / optimal_tour_length
                if info["episode"]["l"] < self.num_cities:
                    info["approximation_ratio"] -= 10
        else:
            info = dict()
        self.previous_state = state

        # convert the state to cost hamiltonian
        pairwise_distances = self.calculate_city_distances(state["coordinates"])
        annotations = state["trajectory"]
        state = np.hstack([pairwise_distances, annotations])

        return state, reward, False, truncate, info

    def calculate_city_distances(self, city_coordinates):
        pairwise_distances = np.zeros(sum(range(self.num_cities)))
        idx = 0
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):  # Only calculate upper triangle
                distance = np.linalg.norm(
                    np.asarray(city_coordinates[i]) - np.asarray(city_coordinates[j])
                )
                # Set distance for both directions
                pairwise_distances[idx] = distance
                idx += 1
        return pairwise_distances

    def tsp_compute_optimal_tour(self, nodes):

        optimal_tour_length = 1000
        for tour_permutation in itertools.permutations(range(1, nodes.shape[0])):
            tour = [0] + list(tour_permutation)
            tour_length = self.tsp_compute_tour_length(nodes, tour)
            if tour_length < optimal_tour_length:
                optimal_tour_length = tour_length
        return optimal_tour_length

    def tsp_compute_tour_length(self, nodes, tour):
        """
        Compute length of a tour, including return to start node.
        (If start node is already added as last node in tour, 0 will be added to tour length.)
        :param nodes: all nodes in the graph in form of (x, y) coordinates
        :param tour: list of node indices denoting a (potentially partial) tour
        :return: tour length
        """
        tour_length = 0
        for i in range(len(tour)):
            if i < len(tour) - 1:
                tour_length += np.linalg.norm(
                    np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i + 1]])
                )
            else:
                tour_length += np.linalg.norm(
                    np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]])
                )

        return tour_length


def make_env(env_id, config):
    def thunk():
        if env_id == "TSP-v1":
            num_cities = config.get("num_cities", 5)
            generator_tsp = UniformGenerator(num_cities=num_cities)
            env = TSP(generator_tsp)
            env = wrappers.JumanjiToGymWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = JumanjiWrapperTSP(env)
            env = ReplayBufferWrapper(env)
        else:
            raise KeyError("This tutorial only works for the TSP problem.")

        return env

    return thunk


def graph_encoding_ansatz(x, input_scaling, weights, wires, layers, num_actions):
    # wmax = max(
    #     np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
    # )  # Normalizing the Hamiltonian is a good idea
    # Apply the initial layer of Hadamard gates to all qubits
    distances = x[:, : sum(range(num_actions))]
    annotations = x[:, sum(range(num_actions)) :]
    annotations_converted = np.zeros_like(annotations, dtype=float)
    # Set values to 0 if negative, π if positive
    annotations_converted[annotations > 0] = np.pi

    for i in wires:
        qml.Hadamard(wires=i)
    # repeat p layers the circuit shown in Fig. 1
    for layer in range(layers):

        idx = 0
        for i in wires:
            for j in range(i + 1, num_actions):
                qml.CNOT(wires=[i, j])
                qml.RZ(input_scaling[layer] * distances[:, idx], wires=j)
                qml.CNOT(wires=[i, j])
                idx += 1

        for i in wires:
            qml.RX(annotations_converted[:, i], wires=i)

    return [qml.expval(qml.PauliX(i)) for i in range(num_actions)]


class DQNAgentQuantum(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        # self.observation_size = np.array(envs.single_observation_space.shape).prod()
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]

        assert (
            self.num_qubits >= self.num_actions
        ), "Number of qubits must be greater than or equal to the number of actions"

        self.num_layers = config["num_layers"]
        self.wires = range(self.num_qubits)

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers), requires_grad=True
        )
        self.output_scaling = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.rand(self.num_layers) * 2 * torch.pi - torch.pi,
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=self.wires)
        self.quantum_circuit = qml.QNode(
            graph_encoding_ansatz,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def forward(self, x):
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.wires,
            self.num_layers,
            self.num_actions,
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
        return logits


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


def dqn_quantum_jumanji(config: dict):
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
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]

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

    q_network = DQNAgentQuantum(envs, config).to(device)
    optimizer = optim.Adam(
        [
            {"params": q_network.input_scaling, "lr": lr_input_scaling},
            {"params": q_network.output_scaling, "lr": lr_output_scaling},
            {"params": q_network.weights, "lr": lr_weights},
        ]
    )
    target_network = DQNAgentQuantum(envs, config).to(device)
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
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
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
        trial_name: str = "dqn_quantum_jumanji"  # Name of the trial
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
        total_timesteps: int = 10000  # Total number of timesteps
        start_e: float = 1.0  # Starting value of epsilon for exploration
        end_e: float = 0.01  # Ending value of epsilon for exploration
        exploration_fraction: float = 0.1  # Fraction of total timesteps for exploration
        learning_starts: int = 1000  # Timesteps before learning starts
        train_frequency: int = 1  # Frequency of training
        batch_size: int = 32  # Batch size for training
        gamma: float = 0.99  # Discount factor
        target_network_frequency: int = 100  # Frequency of target network updates
        tau: float = 0.01  # Soft update coefficient
        lr_input_scaling: float = 0.01  # Learning rate for input scaling
        lr_weights: float = 0.01  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 2  # Number of layers in the quantum circuit
        device: str = "default.qubit"  # Quantum device
        diff_method: str = "backprop"  # Differentiation method
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
    dqn_quantum_jumanji(config)
