# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from replay_buffer import ReplayBuffer, ReplayBufferWrapper


class ArctanNormalizationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


# ENV LOGIC: create you env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ArctanNormalizationWrapper(env)
        env = ReplayBufferWrapper(env)
        return env

    return thunk


# QUANTUM CIRCUIT: define your ansatz here:
def parameterized_quantum_circuit(
    x,
    input_scaling,
    weights,
    num_qubits,
    num_layers,
    num_actions,
    observation_size,
    agent_type,
):
    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RY(input_scaling[layer, i] * x[:, i], wires=[i])
            qml.RZ(input_scaling[layer, i + observation_size] * x[:, i], wires=[i])

        for i in range(num_qubits):
            qml.RZ(weights[layer, i], wires=[i])

        for i in range(num_qubits):
            qml.RY(weights[layer, i + num_qubits], wires=[i])

        if num_qubits == 2:
            qml.CNOT(wires=[0, 1])
        else:
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])

    if agent_type == "actor":
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
    elif agent_type == "critic":
        return [qml.expval(qml.PauliX(0))]


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        self.observation_size = np.array(
            envs.single_observation_space.shape
        ).prod() + np.prod(envs.single_action_space.shape)
        self.num_actions = np.prod(envs.single_action_space.shape)
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]

        assert (
            self.num_qubits >= self.observation_size
        ), "Number of qubits must be greater than or equal to the observation size"
        assert (
            self.num_qubits >= self.num_actions
        ), "Number of qubits must be greater than or equal to the number of actions"

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers, self.num_qubits * 2), requires_grad=True
        )
        self.output_scaling = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.FloatTensor(self.num_layers, self.num_qubits * 2).uniform_(
                -np.pi, np.pi
            ),
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            parameterized_quantum_circuit,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "critic",
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
        return logits


class Actor(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        self.observation_size = np.array(
            envs.single_observation_space.shape
        ).prod()
        self.num_actions = np.prod(envs.single_action_space.shape)
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]

        assert (
            self.num_qubits >= self.observation_size
        ), "Number of qubits must be greater than or equal to the observation size"
        assert (
            self.num_qubits >= self.num_actions
        ), "Number of qubits must be greater than or equal to the number of actions"

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers, self.num_qubits * 2), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.rand(self.num_layers, self.num_qubits * 2) * 2 * torch.pi - torch.pi,
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            parameterized_quantum_circuit,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.action_space.high - envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.action_space.high + envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "actor",
        )
        logits = torch.stack(logits, dim=1)
        return logits * self.action_scale + self.action_bias


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
def ddpg_quantum_continuous_action(config: dict):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    buffer_size = config["buffer_size"]
    total_timesteps = config["total_timesteps"]
    gamma = config["gamma"]
    tau = config["tau"]
    batch_size = config["batch_size"]
    exploration_noise = config["exploration_noise"]
    learning_starts = config["learning_starts"]
    policy_frequency = config["policy_frequency"]
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
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # Here, the quantum agent is initialized with a parameterized quantum circuit
    actor = Actor(envs, config).to(device)
    qf1 = QNetwork(envs, config).to(device)
    qf1_target = QNetwork(envs, config).to(device)
    target_actor = Actor(envs, config).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(
        [
            {"params": qf1.input_scaling, "lr": lr_input_scaling},
            {"params": qf1.output_scaling, "lr": lr_output_scaling},
            {"params": qf1.weights, "lr": lr_weights},
        ]
    )
    actor_optimizer = optim.Adam(
        [
            {"params": actor.input_scaling, "lr": lr_input_scaling},
            {"params": actor.weights, "lr": lr_weights},
        ]
    )

    envs.single_observation_space.dtype = np.float32
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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

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
                    log_metrics(config, metrics, report_path)
        if global_episodes % print_interval == 0 and not ray.is_initialized():
            print(
                "Global step: ", global_step, " Mean return: ", np.mean(episode_returns)
            )

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
            data = rb.sample(batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            if global_step % 100 == 0:
                metrics = {}
                metrics["qf1_values"] = qf1_a_values.mean().item()
                metrics["qf1_loss"] = qf1_loss.item()
                metrics["actor_loss"] = actor_loss.item()
                metrics["SPS"] = int(global_step / (time.time() - start_time))
                log_metrics(config, metrics, report_path)

    if config["save_model"]:
        model_path = f"{os.path.join(report_path, name)}.cleanqrl_model"
        torch.save(actor.state_dict(), model_path)
        torch.save(qf1.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if config["wandb"]:
        wandb.finish()


if __name__ == "__main__":

    @dataclass
    class Config:
        # General parameters
        trial_name: str = "ddpg_quantum_continuous_action"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = True  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "Pendulum-v1"  # Environment ID

        # Algorithm parameters
        total_timesteps: int = 100000  # Total number of timesteps
        buffer_size: int = int(1e6)  # Replay buffer size
        learning_rate: float = 3e-4  # Learning rate
        num_envs: int = 1  # Number of environments
        gamma: float = 0.99  # discount factor
        tau: float = 0.005  # Target network update rate
        batch_size: int = 256  # Batch size
        exploration_noise: float = 0.1  # Std of Gaussian exploration noise
        learning_starts: int = 25e3  # Timesteps before learning starts
        policy_frequency: int = 25  # Frequency of policy updates
        seed: int = None  # Seed for reproducibility
        lr_input_scaling: float = 0.01  # Learning rate for input scaling
        lr_weights: float = 0.01  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 5  # Number of layers in the quantum circuit
        device: str = "lightning.qubit"  # Quantum device
        diff_method: str = "adjoint"  # Differentiation method
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
    ddpg_quantum_continuous_action(config)
