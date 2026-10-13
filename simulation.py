import numpy as np
import pandas as pd

BOLTZMANN_CONSTANT = 1.38e-23


def thermal_noise(temperature, bandwidth):
    return BOLTZMANN_CONSTANT * temperature * bandwidth


def free_space_path_loss_db(distance, wavelength):
    return 20 * np.log10(4 * np.pi * distance / wavelength)


def communication_simulation(parameters, seed=None):
    topology_parameters = parameters["topology"]
    number_of_nodes = topology_parameters["number_of_nodes"]
    radius = topology_parameters["radius"]

    transmission_parameters = parameters["transmission"]
    time_slots = transmission_parameters["time_slots"]
    zone_limits = np.array(transmission_parameters["zone_limits"])
    zone_probabilities = np.array(transmission_parameters["zone_probabilities"])

    channel_parameters = parameters["channel"]
    wavelength = channel_parameters["wavelength"]
    bandwidth = channel_parameters["bandwidth"]
    temperature = channel_parameters["temperature"]
    gamma_threshold = channel_parameters["gamma_threshold"]

    noise = thermal_noise(temperature, bandwidth)

    random_generator = np.random.default_rng(seed)

    distance = radius * random_generator.random(number_of_nodes)

    node_zone_indices = np.digitize(distance, zone_limits)
    node_probabilities = zone_probabilities[node_zone_indices]

    path_loss_db = free_space_path_loss_db(distance, wavelength)
    path_loss = 10 ** (-path_loss_db / 10)

    attempts = np.zeros(number_of_nodes, dtype=int)
    successful_transmissions = np.zeros(number_of_nodes, dtype=int)
    success = np.zeros(number_of_nodes, dtype=bool)

    for slot in range(1, time_slots + 1):
        active_nodes = np.arange(number_of_nodes)
        active_probabilities = node_probabilities[active_nodes]
        random_values = random_generator.random(len(active_nodes))

        transmit = random_values < active_probabilities
        transmitting_nodes = active_nodes[transmit]

        if len(transmitting_nodes) == 0:
            continue

        attempts[transmitting_nodes] += 1

        fading = random_generator.exponential(
            scale=1.0,
            size=len(transmitting_nodes)
        )

        power_at_base_station = fading * path_loss[transmitting_nodes]
        total_power = np.sum(power_at_base_station)

        for i in range(len(transmitting_nodes)):
            node_index = transmitting_nodes[i]
            signal = power_at_base_station[i]
            interference = total_power - signal
            sinr = signal / (noise + interference)

            if sinr >= gamma_threshold:
                success[node_index] = True
                successful_transmissions[node_index] += 1

    results = pd.DataFrame({
        "node": np.arange(number_of_nodes),
        "distance_to_base_station": distance,
        "zone_index": node_zone_indices,
        "transmission_probability": node_probabilities,
        "path_loss_db": path_loss_db,
        "path_loss": path_loss,
        "success": success,
        "attempts": attempts,
        "successful_transmissions": successful_transmissions
    })

    return results
