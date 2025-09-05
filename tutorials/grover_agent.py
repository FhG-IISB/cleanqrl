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
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
import math
# ENV LOGIC: create your env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = gym.make(env_id, is_slippery=config["is_slippery"])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
# QUANTUM CIRCUIT: define your ansatz here:
def oracle(action_vector, num_qubits):
    qml.templates.FlipSign(action_vector, wires=range(num_qubits))
def grover_operator(action, num_qubits):
    binary_action = bin(action)[2:].zfill(num_qubits)
    action_vector = [int(a) for a in binary_action]
    oracle(action_vector, num_qubits)
    qml.templates.GroverOperator(wires=range(num_qubits))
def quantum_circuit(state, grover_circuit_operators, num_qubits):
    # Apply Hadamard to all qubits for superposition
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for action in grover_circuit_operators[state]:
        grover_operator(action, num_qubits)
    return qml.counts(wires=range(num_qubits))
# ALGO LOGIC: initialize your agent here:
class GroverAgentQuantum:
    def __init__(self, state_space, action_space, config):
        self.config = config
        self.state_space = state_space
        self.action_space = action_space
        self.num_qubits = math.ceil(np.log2(self.action_space))
        # optimal number of steps in original Grover's algorithm
        self.max_grover_steps = int(
            round(np.pi / (4 * np.arcsin(1.0 / np.sqrt(2**self.num_qubits))) - 0.5)
        )
        # quality values
        self.state_vals = np.zeros(self.state_space)
        # grover steps taken
        self.num_grover_steps = np.zeros(
            (self.state_space, self.action_space), dtype=int
        )
        # boolean flags to signal maximum amplitude amplification reached
        self.grover_steps_flag = np.zeros(
            (self.state_space, self.action_space), dtype=bool
        )
        self.grover_circuit_operators = [[] for _ in range(self.state_space)]
        self.device = qml.device(config["device"], wires=self.num_qubits, shots=config["shots"])
        self.qnode_actor = qml.QNode(quantum_circuit, self.device)
    def get_action(self, state):
        counts = self.qnode_actor(
            state, self.grover_circuit_operators, self.num_qubits
        )
        max_key = max(counts, key=counts.get)
        action = int(max_key, 2)
        return action
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
def grover_quantum_discrete_action(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, config) for _ in range(num_envs)],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"
    assert isinstance(
        envs.single_observation_space, gym.spaces.Discrete
    ), "only discrete observation space is supported"
    state_space = envs.single_observation_space.n
    action_space = envs.single_action_space.n
    # Here, the quantum agent is initialized with a parameterized quantum circuit
    agent = GroverAgentQuantum(state_space, action_space, config)
    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 50
    episode_returns = deque(maxlen=print_interval)
    circuit_evaluations = 0
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset()
    while global_step < total_timesteps:
        done = False
        # Episode loop
        episode_reward = 0
        episode_length = 0
        while not done:
            state = int(obs[0])
            action = agent.get_action(state)
            circuit_evaluations += 1
            next_obs, reward, terminations, truncations, infos = envs.step(
                np.array([action])
            )
            reward_orig = reward[0]
            episode_reward += reward_orig
            episode_length += 1
            reward = reward_orig * 100
            next_state = int(next_obs[0])
            # update statevals and grover steps
            agent.state_vals[state] += config["alpha"] * (
                reward
                + gamma * agent.state_vals[next_state]
                - agent.state_vals[state]
            )
            steps_num = int(config["k"] * (reward + agent.state_vals[next_state]))
            agent.num_grover_steps[state, action] = min(steps_num, agent.max_grover_steps)
            flag = agent.grover_steps_flag[state, :]
            gsteps = agent.num_grover_steps[state, action]
            if not flag.any():
                for _ in range(gsteps):
                    agent.grover_circuit_operators[state].append(action)
            if gsteps >= agent.max_grover_steps and not flag.any():
                agent.grover_steps_flag[state, action] = True
            global_step += 1
            done = terminations[0] or truncations[0]
            obs = next_obs
        global_episodes += 1
        # If the episode is finished, report the metrics
        # Here addtional logging can be added
        if "episode" in infos:
            for idx, finished in enumerate(infos["_episode"]):
                if finished:
                    metrics = {}
                    episode_returns.append(infos["episode"]["r"].tolist()[idx])
                    metrics["episode_reward"] = infos["episode"]["r"].tolist()[idx]
                    metrics["episode_length"] = infos["episode"]["l"].tolist()[idx]
                    metrics["global_step"] = global_step
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
        # torch.save(agent.state_dict(), model_path)  # no torch params
        print(f"model saved to {model_path}")
    envs.close()
    if config["wandb"]:
        wandb.finish()
if __name__ == "__main__":
    @dataclass
    class Config:
        # General parameters
        trial_name: str = "grover_quantum_discrete_action"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project
        # Environment parameters
        env_id: str = "FrozenLake-v1"  # Environment ID
        # Algorithm parameters
        num_envs: int = 1  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 100000  # Total number of timesteps
        gamma: float = 0.9  # discount factor
        alpha: float = 0.1  # Learning rate for state values
        k: float = 1.0  # Scaling for grover steps
        cuda: bool = False  # Whether to use CUDA
        device: str = "default.qubit"  # Quantum device
        shots: int = 1000  # Shots for quantum circuit
        is_slippery: bool = False  # For FrozenLake
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
    grover_quantum_discrete_action(config)