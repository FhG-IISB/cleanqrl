import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from pathlib import Path
from pennylane import X, Z, I
from jax.scipy.linalg import expm
import pennylane as qml
from pennylane import liealg, math

def test_gradient_speed(config, output_dir="logs/gradient_speed_tests_sum/ki-02"):
    """
    Test the speed of forward pass and gradient computation for quantum circuits
    with varying numbers of qubits (4 to 20), using vmap inside JIT functions.
    Plots results and saves to a single JSON file.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize plot
    plt.ion()  # Interactive mode for live updates
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Circuit Simulation Speed vs. Number of Qubits')
    forward_line, = ax.plot([], [], label='Forward Pass (ms)', marker='o', color='blue')
    gradient_line, = ax.plot([], [], label='Gradient Computation (ms)', marker='s', color='green')
    dim_g_line, = ax2.plot([], [], label='Lie Algebra Dim', marker='o', color='red')
    ax.legend(loc='upper left')
    ax.grid(True)

    ax2.set_ylabel('Lie Algebra Dimension (dim_g)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    # Load existing results if any
    output_path = os.path.join(output_dir, "results.json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            test_results = json.load(f)
    else:
        test_results = []

    # Lists for plotting
    qubit_counts = []
    forward_times = []
    gradient_times = []
    dim_g_list = []
    # Iterate over qubit counts from 4 to 20
    for num_qubits in [4, 6, 8]:
    # for num_qubits in range(4, 25, 2):
        
        # Initialize parameters for the current number of qubits
        num_layers = config["num_layers"]
        params_network = {
            "weights": jnp.array(np.random.uniform(-np.pi, np.pi, size=(num_layers, num_qubits)), dtype=jnp.float32),
            "input_scaling": jnp.ones((num_layers, num_qubits), dtype=jnp.float32),
            "output_scaling": jnp.ones((num_qubits), dtype=jnp.float32),
        }

        # Initialize quantum circuit components
        generators_encoding = [sum([Z(i) for i in range(num_qubits)])]
        generators_variational = [sum([X(i) @ X(i+1) for i in range(num_qubits-1)])]        
        # generators_encoding = [Z(i) for i in range(num_qubits)]
        # generators_variational = [X(i) @ X(i+1) for i in range(num_qubits-1)]
        generators = generators_encoding + generators_variational
        generators_pauli_rep = [op.pauli_rep for op in generators]
        dla = qml.lie_closure(generators_pauli_rep, pauli=True)
        # dim_g = len(dla)
        # adjoint_repr = qml.structure_constants(dla, pauli=True)
        generators_matrix_rep = [op.matrix() for op in generators]
        g = liealg.lie_closure(generators_matrix_rep, matrix=True, verbose=True, tol=1e-5)
        adjoint_repr = liealg.structure_constants(g, matrix=True)
        dim_g = g.shape[0]
        # Prepare the all-zero state
        e_in = np.zeros(dim_g, dtype=float)
        for i, h_i in enumerate(dla):
            rho_in = qml.prod(*(I(i) + Z(i) for i in h_i.wires))
            rho_in = rho_in.pauli_rep
            e_in[i] = (h_i @ rho_in).trace()
        e_in = jnp.array(e_in)

        # Select gates for encoding and variational layers
        gates_encoding = adjoint_repr[:len(generators_encoding)]
        gates_variational = adjoint_repr[len(generators_encoding):]

        # Define measurement operators (assuming num_actions = num_qubits for simplicity)
        num_actions = num_qubits
        measurement_actor = np.zeros((num_actions, dim_g), dtype=float)
        for i in range(num_actions):
            measurement_actor[i, i] = 1.0
        measurement_actor = jnp.array(measurement_actor)

        # Define the circuit function with vmap inside JIT
        @jax.jit
        def gsim_circuit(x, params_network, gates_input_scaling, gates_variational, e_in, w):
            # Ensure x is batched (shape: [batch_size, ])
            x = jnp.atleast_1d(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Convert input to binary (vectorized over batch)
            def convert_to_binary(val):
                bits = (val >> jnp.arange(num_qubits)) & 1
                return bits[::-1]
            x_binary = jax.vmap(convert_to_binary)(x[:, 0])

            # Define single circuit evaluation
            def single_circuit(x_bin):
                e_t = e_in
                for layer in range(num_layers):
                    for i in range(num_qubits):
                        e_t = expm(params_network['input_scaling'][layer, i] * gates_input_scaling[i] * x_bin[i]) @ e_t
                    for i in range(num_qubits-1):
                        e_t = expm(params_network['weights'][layer, i] * gates_variational[i]) @ e_t
                result_g_sim = w @ e_t
                return result_g_sim.real * params_network['output_scaling']

            # Vectorize over batch
            batched_circuit = jax.vmap(single_circuit)(x_binary)
            return batched_circuit

        # Define the loss function
        @jax.jit
        def loss_function_gsim(observations, actions, td_target, params_network, gates_input_scaling, gates_variational, e_in, w):
            # observations: [batch_size, ], actions: [batch_size, 1], td_target: [batch_size, ]
            q_values = gsim_circuit(observations, params_network, gates_input_scaling, gates_variational, e_in, w)
            old_val = q_values[jnp.arange(observations.shape[0]), actions[:, 0]]
            return jnp.mean(jnp.square(td_target - old_val))

        # Generate sample inputs for testing
        batch_size = config["batch_size"]
        observations = jnp.array(np.random.randint(0, 2**num_qubits, size=(batch_size,)), dtype=jnp.int32)
        actions = jnp.array(np.random.randint(0, num_actions, size=(batch_size, 1)), dtype=jnp.int32)
        td_target = jnp.array(np.random.uniform(-1, 1, size=(batch_size,)), dtype=jnp.float32)

        start_time = time.time()

        # Warm-up JIT compilation
        _ = gsim_circuit(observations, params_network, gates_encoding, gates_variational, e_in, measurement_actor)
        _ = jax.value_and_grad(loss_function_gsim, argnums=3)(
            observations, actions, td_target, params_network, gates_encoding, gates_variational, e_in, measurement_actor
        )
        forward_time = (time.time() - start_time) 
        print('compile', forward_time)
        # Measure forward pass time (batched)
        start_time = time.time()
        iterations = 10
        for _ in range(iterations):
            _ = gsim_circuit(observations, params_network, gates_encoding, gates_variational, e_in, measurement_actor)
        forward_time = (time.time() - start_time) / iterations

        # Measure gradient computation time
        start_time = time.time()
        for _ in range(iterations):
            _, grads = jax.value_and_grad(loss_function_gsim, argnums=3)(
                observations, actions, td_target, params_network, gates_encoding, gates_variational, e_in, measurement_actor
            )
        gradient_time = (time.time() - start_time) / iterations

        # Log results
        result = {
            "num_qubits": num_qubits,
            "forward_time_ms": forward_time * 1000,
            "gradient_time_ms": gradient_time * 1000,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "lie_algebra_dim": dim_g
        }
        test_results.append(result)
        print(f"Qubits: {num_qubits}, Forward: {forward_time*1000:.2f} ms, Gradient: {gradient_time*1000:.2f} ms, Dim: {dim_g}")

        # Update plot data
        qubit_counts.append(num_qubits)
        forward_times.append(forward_time * 1000)
        gradient_times.append(gradient_time * 1000)
        dim_g_list.append(dim_g)
        # Update plot
        forward_line.set_xdata(qubit_counts)
        forward_line.set_ydata(forward_times)
        gradient_line.set_xdata(qubit_counts)
        gradient_line.set_ydata(gradient_times)
        dim_g_line.set_xdata(qubit_counts)
        dim_g_line.set_ydata(dim_g_list)

        ax.relim()
        ax.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax.set_yscale('log')  # Use logarithmic scale for better visibility

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.savefig(os.path.join(output_dir, "gradient_speed_plot.png"))

        # Save results to JSON
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Results saved to {output_path}")

    plt.ioff()
    plt.show()
    return test_results

if __name__ == "__main__":
    from dataclasses import dataclass
    import datetime
    import yaml

    @dataclass
    class Config:
        num_layers: int = 4
        batch_size: int = 1

    config = vars(Config())
    test_gradient_speed(config)