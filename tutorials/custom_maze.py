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
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


def optimal_policy_crawford(n):
    optimal_map = ((4,), (3,), (3,), (3, 1), (3,))
    for n in range(n - 2):
        optimal_map += ((1,), (3, 1), tuple(), (1,), (3, 1))
    optimal_map += ((1,), (3, 1), (3,), (1,), (3, 1))
    return optimal_map


def crawford(P, R, n):
    obs = np.zeros((n, 5), dtype=object)
    for n in range(n - 2):
        obs[n + 1, 2] = 'W'
    obs[0, 0] = R
    obs[n - 1, 2] = P
    obs[n - 1, 4] = 'A'
    return obs


def mueller(P, R, **kwargs):
    obs = np.zeros((3, 3), dtype=object)
    obs[0, 0] = R
    x, y = 2, 2
    if x == 0 and y == 0:
        x, y = 2, 2
    obs[y, x] = 'A'
    return obs


def optimal_policy_mueller(**kwargs):
    optimal_map = (
        (4,),
        (3,),
        (3,),
        (1,),
        (3, 1),
        (3, 1),
        (1,),
        (3, 1),
        (3, 1),
    )
    return optimal_map


def neumann_a(P, R, **kwargs):
    obs = np.array([[0, 'A', 0, 0],
                    [0, 'W', 0, P], 
                    [0, 0, 0, R]], dtype=object)
    return obs


def optimal_policy_neumann_a(**kwargs):
    optimal_map = (
        (2, 0),
        (2,),
        (0,),
        (3,),
        (0,),
        tuple(),
        (0,),
        (0,),
        (2,),
        (2,),
        (2,),
        (4,),
    )
    return optimal_map


def neumann_b(P, R, **kwargs):
    obs = np.array([[R, 0, 0, 0, 0], 
                    [0, 0, 'W', 0, 0], 
                    [0, 0, 0, 0, 'A']], dtype=object)
    return obs


def optimal_policy_neumann_b(**kwargs):
    optimal_map = ((4,), (3,), (3,), (3, 1), (3,))
    optimal_map += ((1,), (3, 1), tuple(), (1,), (3, 1))
    optimal_map += ((1,), (3, 1), (3,), (1,), (3, 1))
    return optimal_map


def neumann_c(P, R, **kwargs):
    obs = np.array([[0, 'A',   0,   0,   P], 
                    [0,   0,   P,   0,   P], 
                    [0, 'W', 'W',   R,   0], 
                    [0,   0,   0,   0,   0]], dtype=object)
    return obs


def optimal_policy_neumann_c(**kwargs):
    optimal_map = ((2, 0), (2,), (2,), (0,), (3,))
    optimal_map += ((2, 0), (1,), (2,), (0,), (3, 0))
    optimal_map += ((0,), tuple(), tuple(), (4,), (3,))
    optimal_map += ((2,), (2,), (2,), (1,), (3, 1))
    return optimal_map


maze_types = {
    "crawford": crawford,
    "mueller": mueller,
    "neumann_a": neumann_a,
    "neumann_b": neumann_b,
    "neumann_c": neumann_c,
}

optimal_policy_types = {
    "crawford": optimal_policy_crawford,
    "mueller": optimal_policy_mueller,
    "neumann_a": optimal_policy_neumann_a,
    "neumann_b": optimal_policy_neumann_b,
    "neumann_c": optimal_policy_neumann_c,
}


class CustomMazeEnv(Env):
    def __init__(self, config):
        super(CustomMazeEnv, self).__init__()
        self.config = config
        self.state_encoding = config["state_encoding"]  # binary, onehot
        self.action_array = [0, 1, 2, 3]
        self.action_space = Discrete(4)
        self.maze_name = config["maze_name"]

        self.P = config.get("P", -1)  
        self.R = config.get("R", 1)
        self.N = config.get("N", 0)
        self.n = config.get("n", 3)

        obs = maze_types[self.maze_name](self.P, self.R, n=self.n)
        self.optimal_policy_tuple = optimal_policy_types[self.maze_name](n=self.n)
        self.maze_size_x = obs.shape[1]
        self.maze_size_y = obs.shape[0]
        self.obs_size = int(self.maze_size_y * self.maze_size_x)

        if self.state_encoding == "binary":
            num_bits = (self.maze_size_x * self.maze_size_y).bit_length()
            self.observation_space = Box(-np.inf, np.inf, shape=(num_bits,))
        elif self.state_encoding == "onehot":
            self.observation_space = Box(
                -np.inf, np.inf, shape=(self.maze_size_x * self.maze_size_y,)
            )
        elif self.state_encoding == "integer":
            self.observation_space = Box(
                -np.inf, np.inf, shape=(self.maze_size_x * self.maze_size_y,)
            )

        self.episode = 0


    def reset(self, seed=42, options=None, render=True):
        obs = maze_types[self.maze_name](self.P, self.R, n=self.n)
        self.initial_obs = deepcopy(obs)
        agent_index_y, agent_index_x = np.where(self.initial_obs == 'A')
        y, x = (
            agent_index_y[0],
            agent_index_x[0],
        )  # np.unravel_index(np.argmax(obs), obs.shape)
        self.current_obs = np.zeros(self.initial_obs.shape, dtype=object)
        self.current_obs[y, x] = 1

        if self.state_encoding == "binary":
            index_of_one = np.argmax(np.reshape(self.current_obs, (-1)))  
            num_bits = len(np.reshape(self.current_obs, (-1))).bit_length() 
            binary_rep = format(index_of_one, f"0{num_bits}b")
            state = np.array([int(char) for char in binary_rep], dtype=int)
            return np.reshape(deepcopy(state), -1), {}

        elif self.state_encoding == "onehot":
            return np.reshape(deepcopy(self.current_obs), -1), {}

        elif self.state_encoding == "integer":
            return deepcopy(np.argmax(np.reshape(self.current_obs, (-1)))), {}


    def step(self, action, obs=np.array([None])):
        done = False
        reward = 0
        if not sum(obs == None):
            if self.state_encoding == "binary":
                integer_value = int("".join(map(str, obs.astype(np.int16))), 2)
                self.current_obs = np.zeros(self.obs_size)
                self.current_obs[integer_value] = 1
                self.current_obs = np.reshape(
                    self.current_obs, (self.maze_size_y, self.maze_size_x)
                )
            elif self.state_encoding == "integer":
                integer_value = int(obs)
                obs = np.zeros(self.obs_size)
                obs[integer_value] = 1
                obs = np.reshape(obs, (self.maze_size_y, self.maze_size_x))
            else:
                self.current_obs = deepcopy(
                    np.reshape(obs, (self.maze_size_y, self.maze_size_x))
                )

        agent_index_y, agent_index_x = np.where(self.current_obs == 1)
        y, x = (agent_index_y[0], agent_index_x[0])  
        self.current_obs[y, x] = 0

        if action == 0:
            if (y < self.maze_size_y - 1) and (
                self.initial_obs[y + 1, x] != 'W'
            ):
                y += 1
        elif action == 1:
            if (y > 0) and (self.initial_obs[y - 1, x] != 'W'):
                y -= 1
        elif action == 2:
            if (x < self.maze_size_x - 1) and (
                self.initial_obs[y, x + 1] != 'W'
            ):
                x += 1
        elif action == 3:
            if (x > 0) and (self.initial_obs[y, x - 1] != 'W'):
                x -= 1
        if self.initial_obs[y, x] == self.P:
            reward = self.P
        elif self.initial_obs[y, x] == self.R:
            reward = self.R
            done = True
        else:
            reward = self.N

        self.current_obs[y, x] = 1

        if self.state_encoding == "binary":
            index_of_one = np.argmax(np.reshape(self.current_obs, (-1)))  
            num_bits = len(np.reshape(self.current_obs, (-1))).bit_length() 
            binary_rep = format(index_of_one, f"0{num_bits}b")
            state = np.array([int(char) for char in binary_rep], dtype=int)
            return np.reshape(deepcopy(state), -1), deepcopy(reward), done, False, {}

        elif self.state_encoding == "onehot":
            return (
                deepcopy(np.reshape(self.current_obs, -1)),
                deepcopy(reward),
                done,
                False,
                {},
            )

        elif self.state_encoding == "integer":
            return (
                deepcopy(np.argmax(np.reshape(self.current_obs, (-1)))),
                deepcopy(reward),
                done,
                False,
                {},
            )
        
# ENV LOGIC: create your env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = CustomMazeEnv(config)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


# QUANTUM CIRCUIT: define your ansatz here:
def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
):
    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RX(input_scaling[layer, i] * x[:, i], wires=[i])

        for i in range(num_qubits):
            qml.RZ(weights[layer, i], wires=[i])

        for i in range(num_qubits):
            qml.RY(weights[layer, i + num_qubits], wires=[i])

        if num_qubits == 2:
            qml.CZ(wires=[0, 1])
        else:
            for i in range(num_qubits):
                qml.CZ(wires=[i, (i + 1) % num_qubits])

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]


# ALGO LOGIC: initialize your agent here:
class ReinforceAgentQuantum(nn.Module):
    def __init__(self, observation_size, num_actions, config):
        super().__init__()
        self.config = config
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.softmax = nn.Softmax(dim=-1)

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers, self.num_qubits), requires_grad=True
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

    def forward(self, x):
        # x = self.encode_input(x)
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
        )
        logits = torch.stack(logits, dim=1)
        probs = logits * self.output_scaling
        probs = self.softmax(probs)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)

    def encode_input(self, x):
        x_binary = torch.zeros((x.shape[0], self.observation_size))
        for i, val in enumerate(x):
            binary = bin(int(val.item()))[2:]
            padded = binary.zfill(self.observation_size)
            x_binary[i] = torch.tensor([int(bit) * np.pi for bit in padded])
        return x_binary


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
def reinforce_quantum_discrete_state(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]
    num_qubits = config["num_qubits"]

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
   
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, config) for _ in range(num_envs)],
    )

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # This is for binary state encoding (see tutorials for one hot encoding)
    observation_size = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n

    assert (
        num_qubits >= observation_size
    ), "Number of qubits must be greater than or equal to the observation size"
    assert (
        num_qubits >= num_actions
    ), "Number of qubits must be greater than or equal to the number of actions"

    # Here, the quantum agent is initialized with a parameterized quantum circuit
    agent = ReinforceAgentQuantum(observation_size, num_actions, config).to(device)
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling, "lr": lr_input_scaling},
            {"params": agent.output_scaling, "lr": lr_output_scaling},
            {"params": agent.weights, "lr": lr_weights},
        ]
    )

    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 30
    episode_returns = deque(maxlen=print_interval)
    circuit_evaluations = 0

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset()

    while global_step < total_timesteps:
        log_probs = []
        rewards = []
        done = False

        # Episode loop
        while not done:
            obs = torch.Tensor(obs).to(device)
            action, log_prob = agent.forward(obs)
            obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            rewards.append(reward)
            circuit_evaluations += envs.num_envs
            log_probs.append(log_prob)
            done = np.any(terminations) or np.any(truncations)

        global_step += len(rewards) * num_envs

        # Compute the discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = [
            torch.tensor(Gt, dtype=torch.float32) for Gt in discounted_rewards
        ]

        # Compute the policy loss
        loss = torch.cat(
            [-log_prob * Gt for log_prob, Gt in zip(log_probs, discounted_rewards)]
        ).sum()
        # For each backward pass we need to evaluate the circuit due to the parameter 
        # shift rule at least twice for each parameter on real hardware
        circuit_evaluations += 2*len(rewards)*num_envs*sum([agent.input_scaling.numel(), agent.weights.numel(), agent.output_scaling.numel()])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If the episode is finished, report the metrics
        # Here addtional logging can be added
        if "episode" in infos:
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    metrics = {}
                    global_episodes += 1
                    episode_returns.append(infos["episode"]["r"].tolist()[idx])
                    metrics["episode_reward"] = infos["episode"]["r"].tolist()[idx]
                    metrics["episode_length"] = infos["episode"]["l"].tolist()[idx]
                    metrics["global_step"] = global_step
                    metrics["policy_loss"] = loss.item()
                    metrics["SPS"] = int(global_step / (time.time() - start_time))
                    metrics["circuit_evaluations"] = circuit_evaluations
                    log_metrics(config, metrics, report_path)

            if global_episodes % print_interval == 0 and not ray.is_initialized():
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

    envs.close()
    if config["wandb"]:
        wandb.finish()


if __name__ == "__main__":

    @dataclass
    class Config:
        # General parameters
        trial_name: str = "reinforce_quantum_custom_maze"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "CustomMazeEnv"  # Environment ID
        maze_name: str = "mueller"  # Name of the maze
        state_encoding: str = "binary" # State encoding: binary, onehot, integer
        n: int = 3  # Number of rows in the maze if the maze name is crawford
        P: float = -100 # Value of penalty
        R: float = 100 # Value of reward
        N: float = -10 # Default / neutral reward for all other states

        # Algorithm parameters
        num_envs: int = 2  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 20000  # Total number of timesteps
        gamma: float = 0.95  # discount factor
        lr_input_scaling: float = 0.025  # Learning rate for input scaling
        lr_weights: float = 0.025  # Learning rate for variational parameters
        lr_output_scaling: float = 0.1  # Learning rate for output scaling
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
    reinforce_quantum_discrete_state(config)
