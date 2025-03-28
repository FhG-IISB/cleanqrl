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
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from ray.train._internal.session import get_session
from torch.distributions.categorical import Categorical
from wrapper_jumanji import create_jumanji_env


# ENV LOGIC: create your env (with config) here:
def make_env(env_id, config):
    def thunk():
        env = create_jumanji_env(env_id, config)
        return env

    return thunk


# QUANTUM CIRCUIT: define your ansatz here:
def parametrized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, agent_type
):

    # This block needs to be adapted depending on the environment.
    # The input vector is of shape [4*num_actions] for the Knapsack:
    # [action mask, selected items, values, weights]

    annotations = x[:, num_qubits : num_qubits * 2]
    values_kp = x[:, num_qubits * 2 : num_qubits * 3]
    weights_kp = x[:, 3 * num_qubits :]

    for layer in range(num_layers):
        for block, features in enumerate([annotations, values_kp, weights_kp]):
            for i in range(num_qubits):
                qml.RX(input_scaling[layer, block, i] * features[:, i], wires=[i])

            for i in range(num_qubits):
                qml.RY(weights[layer, block, i], wires=[i])

            for i in range(num_qubits):
                qml.RZ(weights[layer, block, i + num_qubits], wires=[i])

            if num_qubits == 2:
                qml.CZ(wires=[0, 1])
            else:
                for i in range(num_qubits):
                    qml.CZ(wires=[i, (i + 1) % num_qubits])

    if agent_type == "actor":
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
    elif agent_type == "critic":
        return [qml.expval(qml.PauliZ(0))]


# ALGO LOGIC: initialize your agent here:
class PPOAgentQuantumJumanji(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.block_size = 3  # number of subblocks depends on environment

        # input and output scaling are always initialized as ones
        self.input_scaling_critic = nn.Parameter(
            torch.ones(self.num_layers, self.block_size, self.num_qubits),
            requires_grad=True,
        )
        self.output_scaling_critic = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # trainable weights are initialized randomly between -pi and pi
        self.weights_critic = nn.Parameter(
            torch.FloatTensor(
                self.num_layers, self.block_size, self.num_qubits * 2
            ).uniform_(-np.pi, np.pi),
            requires_grad=True,
        )

        # input and output scaling are always initialized as ones
        self.input_scaling_actor = nn.Parameter(
            torch.ones(self.num_layers, self.block_size, self.num_qubits),
            requires_grad=True,
        )
        self.output_scaling_actor = nn.Parameter(
            torch.ones(self.num_actions), requires_grad=True
        )
        # trainable weights are initialized randomly between -pi and pi
        self.weights_actor = nn.Parameter(
            torch.FloatTensor(
                self.num_layers, self.block_size, self.num_qubits * 2
            ).uniform_(-np.pi, np.pi),
            requires_grad=True,
        )

        device = qml.device(config["device"], wires=range(self.num_qubits))
        self.quantum_circuit = qml.QNode(
            parametrized_quantum_circuit,
            device,
            diff_method=config["diff_method"],
            interface="torch",
        )

    def get_value(self, x):
        value = self.quantum_circuit(
            x,
            self.input_scaling_critic,
            self.weights_critic,
            self.num_qubits,
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
            self.num_qubits,
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


# MAIN TRAINING FUNCTION
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
    num_qubits = config["num_qubits"]

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

    num_actions = envs.single_action_space.n

    assert (
        num_qubits >= num_actions
    ), "Number of qubits must be greater than or equal to the number of actions"

    # Here, the quantum agent is initialized with a parameterized quantum circuit
    agent = PPOAgentQuantumJumanji(num_actions, config).to(device)
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
    print_interval = 10
    episode_returns = deque(maxlen=print_interval)
    episode_approximation_ratio = deque(maxlen=print_interval)
    circuit_evaluations = 0

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
                circuit_evaluations += 2*num_envs
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
            circuit_evaluations += num_envs
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
                circuit_evaluations += 2*num_envs*minibatch_size
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
                # For each backward pass we need to evaluate the circuit due to the parameter 
                # shift rule at least twice for each parameter on real hardware
                circuit_evaluations += 2*minibatch_size*sum([agent.input_scaling_actor.numel(), agent.weights_actor.numel(), agent.output_scaling_actor.numel(), agent.input_scaling_critic.numel(), agent.weights_critic.numel(), agent.output_scaling_critic.numel()])
        
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
        metrics["circuit_evaluations"] = circuit_evaluations
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
        total_budget: float = 2

        # Algorithm parameters
        total_timesteps: int = 100000  # Total timesteps for the experiment
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
