
"""
In this script we will show how cleanqrl code can be easily modified to allow for hamiltonian-based QRL, taken from https://arxiv.org/pdf/2405.07790.

The idea is simple: Many combinatorial optimization problems can be formulated as QUBOs. What if we use the QUBO formulation of the problems to inspire
the ansatz of the quantum circuit in QRL? This might improve performance by introducing problem-specific knowledge into the ansatz.

This is an example of how to do this for the Knapsack problem, using the unbalanced penalization method to encode the inequality constraints.
"""

# This file is an adaptation from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
import torch.optim as optim
import wandb
import yaml
from jumanji import wrappers
from jumanji.environments import Knapsack
from jumanji.environments.packing.knapsack.generator import RandomGenerator
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical

"""
The first step is to write a new wrapper for the Knapsack environment from jumanji to return the observation as a cost Hamiltonian of the problem.

This requires defining two functions: ```formulate_knapsack_qubo_unbalanced``` and ```convert_QUBO_to_ising```.

The first function takes the weights and values of the items and the total budget, and returns the QUBO formulation of the problem-
the second function takes the QUBO matrix and returns the Ising Hamiltonian, which is then passed to the agent to generate the quantum circuit.

In this initial class we just rewrite the ```init```, ```reset``` and ```step``` methods of the original class.
"""


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
            generator_knapsack = RandomGenerator(
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

"""
The ansatz of this script is also different. In particular, it is based on the cost Hamiltonian of the problem.

"""

def cost_hamiltonian_ansatz(
    x, input_scaling, weights, wires, layers, num_actions, agent_type
):
    # wmax = max(
    #     np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
    # )  # Normalizing the Hamiltonian is a good idea
    # Apply the initial layer of Hadamard gates to all qubits
    h = x[:, :num_actions]
    J = x[:, num_actions:]

    for i in wires:
        qml.Hadamard(wires=i)
    # repeat p layers the circuit shown in Fig. 1
    for layer in range(layers):
        # ---------- COST HAMILTONIAN ----------
        for idx_h in wires:  # single-qubit terms
            qml.RZ(input_scaling[layer] * h[:, idx_h], wires=idx_h)

        idx_j = 0
        for i in wires:
            for j in range(i + 1, num_actions):
                qml.CNOT(wires=[i, j])
                qml.RZ(input_scaling[layer] * J[:, idx_j], wires=j)
                qml.CNOT(wires=[i, j])
                idx_j += 1
        # ---------- MIXER HAMILTONIAN ----------
        for i in wires:
            qml.RX(weights[layer], wires=i)

    if agent_type == "actor":
        return [qml.expval(qml.PauliZ(i)) for i in range(num_actions)]
    elif agent_type == "critic":
        return [qml.expval(qml.PauliZ(0))]
    
"""
The remainder of the code is unchanged from the original ```ppo_quantum_jumanji.py``` implementation.
"""

class PPOAgentQuantumJumanji(nn.Module):
    def __init__(self, envs, config):
        super().__init__()
        self.config = config
        self.envs = envs
        # observation size?
        self.num_actions = envs.single_action_space.n
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.wires = range(self.num_qubits)

        assert (
            self.num_qubits >= self.num_actions
        ), "Number of qubits must be greater than or equal to the number of actions"

        # input and output scaling are always initialized as ones
        self.input_scaling_critic = nn.Parameter(
            torch.ones(self.num_layers), requires_grad=True
        )
        self.output_scaling_critic = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # trainable weights are initialized randomly between -pi and pi
        self.weights_critic = nn.Parameter(
            torch.rand(self.num_layers) * 2 * torch.pi - torch.pi,
            requires_grad=True,
        )

        # input and output scaling are always initialized as ones
        self.input_scaling_actor = nn.Parameter(
            torch.ones(self.num_layers), requires_grad=True
        )
        self.output_scaling_actor = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights_actor = nn.Parameter(
            torch.rand(self.num_layers) * 2 * torch.pi - torch.pi,
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=self.wires)
        self.quantum_circuit = qml.QNode(
            cost_hamiltonian_ansatz,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def get_value(self, x):
        value = self.quantum_circuit(
            x,
            self.input_scaling_critic,
            self.weights_critic,
            self.wires,
            self.num_layers,
            self.num_actions,
            "critic",
        )
        value = torch.stack(value, dim=1)
        value = value * self.output_scaling_critic
        return value

    def get_action_and_value(self, x, action=None):
        logits = self.quantum_circuit(
            x,
            self.input_scaling_actor,
            self.weights_actor,
            self.wires,
            self.num_layers,
            self.num_actions,
            "actor",
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling_actor
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


def log_metrics(config, metrics, report_path=None):
    if config["wandb"]:
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        with open(os.path.join(report_path, "result.json"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def ppo_quantum_jumanji(config):
    num_envs = config["num_envs"]
    num_steps = config["num_steps"]
    num_minibatches = config["num_minibatches"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    anneal_lr = config["anneal_lr"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]
    gamma = config["gamma"]
    gae_lambda = config["gae_lambda"]
    update_epochs = config["update_epochs"]
    clip_coef = config["clip_coef"]
    norm_adv = config["norm_adv"]
    clip_vloss = config["clip_vloss"]
    ent_coef = config["ent_coef"]
    vf_coef = config["vf_coef"]
    target_kl = config["target_kl"]
    max_grad_norm = config["max_grad_norm"]
    cuda = config["cuda"]

    if target_kl == "None":
        target_kl = None

    if config["seed"] == "None":
        config["seed"] = None

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

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

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, config) for i in range(num_envs)],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = PPOAgentQuantumJumanji(envs, config).to(device)
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling_actor, "lr": lr_input_scaling},
            {"params": agent.output_scaling_actor, "lr": lr_output_scaling},
            {"params": agent.weights_actor, "lr": lr_weights},
            {"params": agent.input_scaling_critic, "lr": lr_input_scaling},
            {"params": agent.output_scaling_critic, "lr": lr_output_scaling},
            {"params": agent.weights_critic, "lr": lr_weights},
        ]
    )

    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # global parameters to log
    global_step = 0
    global_episodes = 0
    print_interval = 3
    episode_returns = deque(maxlen=print_interval)
    episode_approximation_ratio = deque(maxlen=print_interval)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            for idx, param_group in enumerate(optimizer.param_groups):
                previous_lr = param_group["lr"]
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * previous_lr
                optimizer.param_groups[idx]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

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
                        if "approximation_ratio" in infos.keys():
                            metrics["approximation_ratio"] = infos[
                                "approximation_ratio"
                            ][idx]
                            episode_approximation_ratio.append(
                                metrics["approximation_ratio"]
                            )
                        log_metrics(config, metrics, report_path)

                if global_episodes % print_interval == 0 and not ray.is_initialized():
                    logging_info = f"Global step: {global_step}  Mean return: {np.mean(episode_returns)}"
                    if len(episode_approximation_ratio) > 0:
                        logging_info += f"  Mean approximation ratio: {np.mean(episode_approximation_ratio)}"
                    print(logging_info)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        metrics = {}
        metrics["lr_input_scaling"] = optimizer.param_groups[0]["lr"]
        metrics["lr_output_scaling"] = optimizer.param_groups[1]["lr"]
        metrics["lr_weights"] = optimizer.param_groups[2]["lr"]
        metrics["value_loss"] = v_loss.item()
        metrics["policy_loss"] = pg_loss.item()
        metrics["entropy"] = entropy_loss.item()
        metrics["old_approx_kl"] = old_approx_kl.item()
        metrics["approx_kl"] = approx_kl.item()
        metrics["clipfrac"] = np.mean(clipfracs)
        metrics["explained_variance"] = np.mean(explained_var)
        metrics["SPS"] = int(global_step / (time.time() - start_time))
        log_metrics(config, metrics, report_path)

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
        trial_name: str = "ppo_quantum_jumanji"  # Name of the trial
        trial_path: str = "logs"  # Path to save logs relative to the parent directory
        wandb: bool = False  # Use wandb to log experiment data
        project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

        # Environment parameters
        env_id: str = "Knapsack-v1"  # Environment ID
        num_items: int = 4
        total_budget: int = 2

        # Algorithm parameters
        total_timesteps: int = 1000000  # Total timesteps for the experiment
        num_envs: int = 1  # Number of parallel environments
        seed: int = None  # Seed for reproducibility
        num_steps: int = 2048  # Steps per environment per policy rollout
        anneal_lr: bool = True  # Toggle for learning rate annealing
        lr_input_scaling: float = 0.01  # Learning rate for input scaling
        lr_weights: float = 0.01  # Learning rate for variational parameters
        lr_output_scaling: float = 0.01  # Learning rate for output scaling
        gamma: float = 0.9  # Discount factor gamma
        gae_lambda: float = 0.95  # Lambda for general advantage estimation
        num_minibatches: int = 32  # Number of mini-batches
        update_epochs: int = 10  # Number of epochs to update the policy
        norm_adv: bool = True  # Toggle for advantages normalization
        clip_coef: float = 0.2  # Surrogate clipping coefficient
        clip_vloss: bool = True  # Toggle for clipped value function loss
        ent_coef: float = 0.0  # Entropy coefficient
        vf_coef: float = 0.5  # Value function coefficient
        max_grad_norm: float = 0.5  # Maximum gradient norm for clipping
        target_kl: float = None  # Target KL divergence threshold
        cuda: bool = False  # Whether to use CUDA
        num_qubits: int = 4  # Number of qubits
        num_layers: int = 2  # Number of layers in the quantum circuit
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

    ppo_quantum_jumanji(config)
