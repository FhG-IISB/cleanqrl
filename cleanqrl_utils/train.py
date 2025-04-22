import os
import sys

# This is important for the import of the cleanqrl package. Do not delete this line
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "cleanqrl"))

from cleanqrl.ddpg_classical_continuous_action import ddpg_classical_continuous_action
from cleanqrl.ddpg_quantum_continuous_action import ddpg_quantum_continuous_action

# DQN classical
from cleanqrl.dqn_classical import dqn_classical
from cleanqrl.dqn_classical_discrete_state import dqn_classical_discrete_state
from cleanqrl.dqn_classical_jumanji import dqn_classical_jumanji

# DQN quantum
from cleanqrl.dqn_quantum import dqn_quantum
from cleanqrl.dqn_quantum_discrete_state import dqn_quantum_discrete_state
from cleanqrl.dqn_quantum_jumanji import dqn_quantum_jumanji

# PPO classical
from cleanqrl.ppo_classical import ppo_classical
from cleanqrl.ppo_classical_continuous_action import ppo_classical_continuous_action
from cleanqrl.ppo_classical_discrete_state import ppo_classical_discrete_state
from cleanqrl.ppo_classical_jumanji import ppo_classical_jumanji

# PPO quantum
from cleanqrl.ppo_quantum import ppo_quantum
from cleanqrl.ppo_quantum_continuous_action import ppo_quantum_continuous_action
from cleanqrl.ppo_quantum_discrete_state import ppo_quantum_discrete_state
from cleanqrl.ppo_quantum_jumanji import ppo_quantum_jumanji

# REINFORCE classical
from cleanqrl.reinforce_classical import reinforce_classical
from cleanqrl.reinforce_classical_continuous_action import (
    reinforce_classical_continuous_action,
)
from cleanqrl.reinforce_classical_discrete_state import (
    reinforce_classical_discrete_state,
)
from cleanqrl.reinforce_classical_jumanji import reinforce_classical_jumanji

# REINFORCE quantum
from cleanqrl.reinforce_quantum import reinforce_quantum
from cleanqrl.reinforce_quantum_continuous_action import (
    reinforce_quantum_continuous_action,
)
from cleanqrl.reinforce_quantum_discrete_state import reinforce_quantum_discrete_state
from cleanqrl.reinforce_quantum_jumanji import reinforce_quantum_jumanji

from tutorials.graph_encoding import reinforce_quantum_jumanji as reinforce_quantum_jumanji_graph_encoding
from tutorials.hamiltonian_encoding import reinforce_quantum_jumanji as reinforce_quantum_jumanji_hamiltonian_encoding
from tutorials.noise_models import reinforce_quantum_discrete_state as reinforce_quantum_discrete_state_noise_model
from tutorials.custom_maze import reinforce_quantum_discrete_state as reinforce_quantum_discrete_state_custom_maze
agent_switch = {
    "ppo_classical": ppo_classical,
    "ppo_classical_continuous_action": ppo_classical_continuous_action,
    "ppo_classical_discrete_state": ppo_classical_discrete_state,
    "ppo_classical_jumanji": ppo_classical_jumanji,
    "ppo_quantum": ppo_quantum,
    "ppo_quantum_continuous_action": ppo_quantum_continuous_action,
    "ppo_quantum_discrete_state": ppo_quantum_discrete_state,
    "ppo_quantum_jumanji": ppo_quantum_jumanji,
    "dqn_classical": dqn_classical,
    "dqn_classical_discrete_state": dqn_classical_discrete_state,
    "dqn_classical_jumanji": dqn_classical_jumanji,
    "dqn_quantum": dqn_quantum,
    "dqn_quantum_discrete_state": dqn_quantum_discrete_state,
    "dqn_quantum_jumanji": dqn_quantum_jumanji,
    "ddpg_classical_continuous_action": ddpg_classical_continuous_action,
    "ddpg_quantum_continuous_action": ddpg_quantum_continuous_action,
    "reinforce_classical": reinforce_classical,
    "reinforce_classical_continuous_action": reinforce_classical_continuous_action,
    "reinforce_classical_discrete_state": reinforce_classical_discrete_state,
    "reinforce_classical_jumanji": reinforce_classical_jumanji,
    "reinforce_quantum": reinforce_quantum,
    "reinforce_quantum_continuous_action": reinforce_quantum_continuous_action,
    "reinforce_quantum_discrete_state": reinforce_quantum_discrete_state,
    "reinforce_quantum_jumanji": reinforce_quantum_jumanji,
    "reinforce_quantum_jumanji_graph_encoding": reinforce_quantum_jumanji_graph_encoding,
    "reinforce_quantum_jumanji_hamiltonian_encoding": reinforce_quantum_jumanji_hamiltonian_encoding,
    "reinforce_quantum_discrete_state_noise_model": reinforce_quantum_discrete_state_noise_model,
    "reinforce_quantum_discrete_state_custom_maze": reinforce_quantum_discrete_state_custom_maze,
}

def train_agent(config):
    try:
        agent_type = config["agent"].lower()
        if agent_type not in agent_switch:
            raise KeyError(
                f"Agent type '{agent_type}' not found in agent_switch dictionary"
            )
        agent_switch[agent_type](config)
    except KeyError as e:
        raise e
