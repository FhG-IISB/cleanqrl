import pennylane as qml
import torch
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt

def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
):
    
    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RZ(2*input_scaling[layer, i] * x[i], wires=[i])
        for i in range(num_qubits-1):
            qml.IsingXX(2*weights[layer, i], wires=[i, (i + 1) % num_qubits])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]

def test_pennylane_speed(config, output_dir="logs/pennylane_speed_tests_5/ki-02"):
    """
    Test the speed of forward pass and gradient computation for PennyLane quantum circuits
    with varying numbers of qubits (4 to 20), using PyTorch-style gradient computation.
    Processes observations individually in a for loop and computes loss directly.
    Plots results and saves to a single JSON file.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize plot
    plt.ion()  # Interactive mode for live updates
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (ms)')
    ax.set_title('PennyLane Circuit Simulation Speed vs. Number of Qubits')
    forward_line, = ax.plot([], [], label='Forward Pass (ms)', marker='o')
    gradient_line, = ax.plot([], [], label='Gradient Computation (ms)', marker='s')
    ax.legend()
    ax.grid(True)

    # Load existing results if any
    output_path = os.path.join(output_dir, "results_tensor.json")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            test_results = json.load(f)
    else:
        test_results = []

    # Lists for plotting
    qubit_counts = []
    forward_times = []
    gradient_times = []
   
    # Iterate over qubit counts from 4 to 20
    for num_qubits in range(4, 25, 2):
        num_layers = config["num_layers"]
        batch_size = config["batch_size"]
        observation_size = num_qubits
        num_actions = num_qubits
        dev = qml.device("default.tensor", wires=num_qubits)

        quantum_circuit = qml.QNode(
            parameterized_quantum_circuit,
            dev,
            diff_method='backprop',
            interface="torch",
        )
        # Initialize parameters
        input_scaling = torch.randn(num_layers, observation_size, requires_grad=True)
        weights = torch.randn(num_layers, num_qubits-1, requires_grad=True)
        optimizer = torch.optim.Adam([input_scaling, weights], lr=0.01)

        # Generate sample inputs
        observations = torch.randint(0, 2, (batch_size, observation_size), dtype=torch.float32)
        actions = torch.randint(0, num_actions, (batch_size,))
        td_targets = torch.randn(batch_size)

        # Warm-up
        for i in range(batch_size):
            _ = quantum_circuit(
                observations[i], input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
            )
        
        # Measure forward pass time
        start_time = time.time()
        iterations = 10
        for _ in range(iterations):
            for i in range(batch_size):
                q_values = quantum_circuit(
                    observations[i], input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
                )
        forward_time = (time.time() - start_time) / iterations

        # Measure gradient computation time
        start_time = time.time()
        for _ in range(iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            for i in range(batch_size):
                q_values = quantum_circuit(
                    observations[i], input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
                )
                selected_q = q_values[actions[i]]
                loss = (td_targets[i] - selected_q) ** 2
                total_loss += loss
            total_loss = total_loss / batch_size
            total_loss.backward()
            optimizer.step()
        gradient_time = (time.time() - start_time) / iterations

        # Log results
        result = {
            "num_qubits": num_qubits,
            "forward_time_ms": forward_time * 1000,
            "gradient_time_ms": gradient_time * 1000,
            "num_layers": num_layers,
            "batch_size": batch_size
        }
        test_results.append(result)
        print(f"Qubits: {num_qubits}, Forward: {forward_time*1000:.2f} ms, Gradient: {gradient_time*1000:.2f} ms")

        # Update plot data
        qubit_counts.append(num_qubits)
        forward_times.append(forward_time * 1000)
        gradient_times.append(gradient_time * 1000)

        # Update plot
        forward_line.set_xdata(qubit_counts)
        forward_line.set_ydata(forward_times)
        gradient_line.set_xdata(qubit_counts)
        gradient_line.set_ydata(gradient_times)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.savefig(os.path.join(output_dir, "pennylane_speed_plot_tensor.png"))

        # Save results to JSON
        with open(output_path, "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Results saved to {output_path}")

    plt.ioff()
    plt.show()
    return test_results

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        num_layers: int = 4
        batch_size: int = 1

    config = vars(Config())
    test_pennylane_speed(config)