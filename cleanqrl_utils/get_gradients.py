# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import random
import time

import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ContinuousEncoding(env)
        return env

    return thunk


class ContinuousEncoding(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def get_gradients(config):
    cuda = config["cuda"]
    env_id = config["env_id"]
    num_envs = config["num_envs"]
    learning_rate = config["learning_rate"]
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
    lr_variational = config["lr_variational"]
    lr_output_scaling = config["lr_output_scaling"]
    num_iterations = config["num_iterations"]

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
            )
            for i in range(num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    for iteration in range(num_iterations):
        q_network = QRLAgentDQN(envs, config).to(device)
        optimizer = optim.Adam(
            [
                {
                    "params": q_network._parameters["input_scaling_actor"],
                    "lr": lr_input_scaling,
                },
                {
                    "params": q_network._parameters["output_scaling_actor"],
                    "lr": lr_output_scaling,
                },
                {
                    "params": q_network._parameters["variational_actor"],
                    "lr": lr_variational,
                },
            ]
        )
        target_network = QRLAgentDQN(envs, config).to(device)
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        global_episodes = 0
        episode_returns = []
        global_step_returns = []

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset()
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
                        target_max, _ = target_network(data.next_observations).max(
                            dim=1
                        )
                        td_target = data.rewards.flatten() + gamma * target_max * (
                            1 - data.dones.flatten()
                        )
                    old_val = (
                        q_network(data.observations).gather(1, data.actions).squeeze()
                    )
                    loss = F.mse_loss(td_target, old_val)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()

                    grads = {}
                    for name, param in q_network.named_parameters():
                        if param.grad is not None:
                            grads[name] = param.grad.detach().numpy().flatten()

                    ray.train.report(metrics={"gradients": grads})
                    break

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

    envs.close()
