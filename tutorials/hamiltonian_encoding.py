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
from wrapper_jumanji import create_jumanji_env

import sympy as sp
import gymnasium as gym
import jumanji
import jumanji.wrappers as wrappers
import numpy as np
from jumanji.environments import TSP, Knapsack, Maze
from jumanji.environments.packing.knapsack.generator import (
    RandomGenerator as RandomGeneratorKnapsack,
)


# We need to create a new wrapper for the TSP environment that returns
# the observation into a cost hamiltonian of the problem.
class JumanjiWrapperKnapsack(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # For the knapsack problem we use the so called unbalanced penalization method
        # This means that we will have sum(range(num_items)) quadratic terms + num_items linear terms
        # This is constant throughout
        self.num_items = self.env.unwrapped.num_items
        self.total_budget = self.env.unwrapped.total_budget
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(sum(range(self.num_items)) + self.num_items,)
        )

    def reset(self, **kwargs):
        state, info = self.env.reset()
        # convert the state to cost hamiltonian
        offset, QUBO = self.formulate_knapsack_qubo_unbalanced(
            state["weights"], state["values"], self.total_budget
        )
        offset, h, J = self.convert_QUBO_to_ising(offset, QUBO)
        state = np.hstack([h, J])
        return state, info

    def step(self, action):
        state, reward, terminate, truncate, info = self.env.step(action)
        if truncate:
            values = self.previous_state["values"]
            weights = self.previous_state["weights"]
            optimal_value = self.knapsack_optimal_value(
                weights, values, self.total_budget
            )
            info["optimal_value"] = optimal_value
            info["approximation_ratio"] = info["episode"]["r"] / optimal_value
            # if info['approximation_ratio'] > 0.9:
            #     print(info['approximation_ratio'])
        else:
            info = dict()
        self.previous_state = state

        # convert the state to cost hamiltonian
        offset, QUBO = self.formulate_knapsack_qubo_unbalanced(
            state["weights"], state["values"], self.total_budget
        )
        offset, h, J = self.convert_QUBO_to_ising(offset, QUBO)
        state = np.hstack([h, J])

        return state, reward, False, truncate, info
    
    """
    We also define the knapsack_optimal_value function, which is used to calculate the optimal value of the knapsack problem.
    This then allows us to claculate the approximation ratio of the solutions found by the agent.
    """

    def knapsack_optimal_value(self, weights, values, total_budget, precision=1000):
        """
        Solves the knapsack problem with float weights and values between 0 and 1.

        Args:
            weights: List or array of item weights (floats between 0 and 1)
            values: List or array of item values (floats between 0 and 1)
            capacity: Maximum weight capacity of the knapsack (float)
            precision: Number of discretization steps for weights (default: 1000)

        Returns:
            The maximum value that can be achieved
        """
        # Convert to numpy arrays
        weights = np.array(weights)
        values = np.array(values)

        # Validate input
        if not np.all((0 <= weights) & (weights <= 1)) or not np.all(
            (0 <= values) & (values <= 1)
        ):
            raise ValueError("All weights and values must be between 0 and 1")

        if total_budget < 0:
            raise ValueError("Capacity must be non-negative")

        n = len(weights)
        if n == 0:
            return 0.0

        # Scale weights to integers for dynamic programming
        scaled_weights = np.round(weights * precision).astype(int)
        scaled_capacity = int(total_budget * precision)

        # Initialize DP table
        dp = np.zeros(scaled_capacity + 1)

        # Fill the DP table
        for i in range(n):
            # We need to go backward to avoid counting an item multiple times
            for w in range(scaled_capacity, scaled_weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - scaled_weights[i]] + values[i])

        return float(dp[scaled_capacity])
    
    """
    This function is extremely important. As mentioned before, it formulates the problem as a QUBO.

    This is mostly trivial, were it not for the inequality constraints. These, unlike other constraints, are harder to encode into QUBOs that,
    by definition, are unconstrained. There are several ways of doing this. The most common one requires the use of slack variables, which
    correspond to extra qubits in the QUBO. However, here we use the unbalanced penalization method from https://arxiv.org/pdf/2211.13914.
    This method encodes the inequality constraints as the second-degree Taylor expansion of the exponential decay function. Effectively, it
    requires no additional qubits, but the optimal solution may no longer be the global minimum of the QUBO.
    """

    def formulate_knapsack_qubo_unbalanced(
        self, weights, values, total_budget, lambdas=None
    ):
        """
        Formulates the QUBO with the unbalanced penalization method.
        This means the QUBO does not use additional slack variables.
        Params:
            lambdas: Correspond to the penalty factors in the unbalanced formulation.
        """
        if lambdas is None:
            lambdas = [0.96, 0.0371]
        num_items = len(values)
        x = [sp.symbols(f"{i}") for i in range(num_items)]
        cost = 0
        constraint = 0

        for i in range(num_items):
            cost -= x[i] * values[i]
            constraint += x[i] * weights[i]

        H_constraint = total_budget - constraint
        H_constraint_taylor = (
            1 - lambdas[0] * H_constraint + 0.5 * lambdas[1] * H_constraint**2
        )
        H_total = cost + H_constraint_taylor
        H_total = H_total.expand()
        H_total = sp.simplify(H_total)

        for i in range(len(x)):
            H_total = H_total.subs(x[i] ** 2, x[i])

        H_total = H_total.expand()
        H_total = sp.simplify(H_total)

        "Transform into QUBO matrix"
        coefficients = H_total.as_coefficients_dict()

        # Remove the offset
        try:
            offset = coefficients.pop(1)
        except IndexError:
            print("Warning: No offset found in coefficients. Using default of 0.")
            offset = 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            offset = 0

        # Get the QUBO
        QUBO = np.zeros((num_items, num_items))
        for key, value in coefficients.items():
            key = str(key)
            parts = key.split("*")
            if len(parts) == 1:
                QUBO[int(parts[0]), int(parts[0])] = value
            elif len(parts) == 2:
                QUBO[int(parts[0]), int(parts[1])] = value / 2
                QUBO[int(parts[1]), int(parts[0])] = value / 2
        return offset, QUBO
    
    """
    This next function simply converts the QUBO matrix into the Ising Hamiltonian.
    """

    def convert_QUBO_to_ising(self, offset, Q):
        """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
        n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
        # Create default dictionaries to store h and pairwise interactions J
        h = np.zeros(Q.shape[0])
        J = np.zeros(sum(range(Q.shape[0])))
        idj = 0
        # Loop over each qubit (variable) in the QUBO matrix
        for i in range(n_qubits):
            # Update the magnetic field for qubit i based on its diagonal element in Q
            h[i] -= Q[i, i] / 2
            # Update the offset based on the diagonal element in Q
            offset += Q[i, i] / 2
            # Loop over other qubits (variables) to calculate pairwise interactions
            for j in range(i + 1, n_qubits):
                # Update the pairwise interaction strength (J) between qubits i and j
                J[idj] = Q[i, j] / 4
                # Update the magnetic fields for qubits i and j based on their interactions in Q
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                # Update the offset based on the interaction strength between qubits i and j
                offset += Q[i, j] / 4
                idj += 1

        return offset, h, J


def make_env(env_id, config):
    def thunk():
        if env_id == "Knapsack-v1":
            num_items = config.get("num_items", 5)
            total_budget = config.get("total_budget", 2)
            generator_knapsack = RandomGeneratorKnapsack(
                num_items=num_items, total_budget=total_budget
            )
            env = Knapsack(generator=generator_knapsack)
            env = wrappers.JumanjiToGymWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = JumanjiWrapperKnapsack(env)
        else:
            raise KeyError("This tutorial only works for the Knapsack problem.")

        return env

    return thunk

# QUANTUM CIRCUIT: This function contains the key differenz to the standard approach
def hamiltonian_encoding_ansatz(x, input_scaling, weights, num_qubits, num_layers, num_actions):

    annotations_mask = x[:, :num_actions]
    annotations = torch.zeros_like(annotations_mask, dtype=float)
    # Set values to 0 if negative, π if positive
    annotations[annotations_mask > 0] = torch.pi

    h = x[:, :num_actions]
    J = x[:, num_actions:]

    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # repeat p layers the circuit shown in Fig. 1
    for layer in range(num_layers):
        # ---------- COST HAMILTONIAN ----------
        for idx_h in range(num_qubits):  # single-qubit terms
            qml.RZ(input_scaling[layer] * h[:, idx_h], wires=idx_h)

        idx_J = 0
        for i in range(num_qubits):
            for j in range(i + 1, num_actions):
                qml.CNOT(wires=[i, j])
                qml.RZ(input_scaling[layer] * J[:, idx_J], wires=j)
                qml.CNOT(wires=[i, j])
                idx_J += 1

        # ---------- MIXER HAMILTONIAN ----------
        for i in range(num_qubits):
            qml.RX(weights[layer]*annotations[:, i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(num_actions)]


# ALGO LOGIC: initialize your agent here:
class ReinforceAgentQuantum(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]

        # input and output scaling are always initialized as ones
        self.input_scaling = nn.Parameter(
            torch.ones(self.num_layers),
            requires_grad=True,
        )
        self.output_scaling = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights = nn.Parameter(
            torch.FloatTensor(
                self.num_layers
            ).uniform_(-np.pi, np.pi),
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            hamiltonian_encoding_ansatz,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def get_action_and_logprob(self, x):
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action)


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
def reinforce_quantum_jumanji(config):
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

    num_actions = envs.single_action_space.n

    assert (
        num_qubits >= num_actions
    ), "Number of qubits must be greater than or equal to the number of actions"

    # Here, the quantum agent is initialized with a parameterized quantum circuit
    agent = ReinforceAgentQuantum(num_actions, config).to(device)
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
    print_interval = 50
    episode_returns = deque(maxlen=print_interval)
    episode_approximation_ratio = deque(maxlen=print_interval)
    circuit_evaluations = 0

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(device)

    while global_step < total_timesteps:
        log_probs = []
        rewards = []
        done = False

        # Episode loop
        while not done:
            action, log_prob = agent.get_action_and_logprob(obs)
            log_probs.append(log_prob)
            obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            rewards.append(reward)
            circuit_evaluations += envs.num_envs
            obs = torch.Tensor(obs).to(device)
            done = np.any(terminations) or np.any(truncations)

        global_episodes += 1

        # Not sure about this?
        global_step += len(rewards) * num_envs

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(np.array(discounted_rewards)).to(device)

        # Calculate policy gradient loss
        loss = torch.cat(
            [-log_prob * Gt for log_prob, Gt in zip(log_probs, discounted_rewards)]
        ).sum()
        # For each backward pass we need to evaluate the circuit due to the parameter 
        # shift rule at least twice for each parameter on real hardware
        circuit_evaluations += 2*len(rewards)*num_envs*sum([agent.input_scaling.numel(), agent.weights.numel(), agent.output_scaling.numel()])
        
        # Update the policy
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
                    metrics["circuit_evaluations"] = circuit_evaluations
                    metrics["SPS"] = int(global_step / (time.time() - start_time))
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
        trial_name: str = "reinforce_quantum_jumanji"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "Knapsack-v1"  # Environment ID
        num_items: int = 3
        total_budget: float = 1.5

        # Algorithm parameters
        num_envs: int = 1  # Number of environments
        seed: int = None  # Seed for reproducibility
        total_timesteps: int = 100000  # Total number of timesteps
        gamma: float = 0.99  # discount factor
        lr_input_scaling: float = 0.001  # Learning rate for input scaling
        lr_weights: float = 0.001  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 3  # Number of qubits
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
    reinforce_quantum_jumanji(config)
