
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def communication_simulation(parameters):

    # -------------------------------
    # 1. Parameters
    # -------------------------------

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

    # Θερμικός θόρυβος: N = k * T * B
    boltzmann_constant = 1.38 * 10**(-23)
    noise = boltzmann_constant * temperature * bandwidth

    # -------------------------------
    # 2. Random generator
    # -------------------------------

    random_generator = np.random.default_rng(seed=42)

    # -------------------------------
    # 3. Random node positions
    # -------------------------------

    # Ομοιόμορφη τυχαία κατανομή μέσα στον κύκλο
    distance = radius * np.sqrt(
        random_generator.random(number_of_nodes)
    )

    angle = 2 * np.pi * random_generator.random(number_of_nodes)

    # Συντεταγμένες των κόμβων
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)

    # -------------------------------
    # 4. Transmission probability by zone
    # -------------------------------

    node_zone_indices = np.digitize(distance, zone_limits)

    node_probabilities = zone_probabilities[node_zone_indices]

    # -------------------------------
    # 5. Free-space path loss
    # -------------------------------

    # PL(dB) = 20 * log10(4 * π * d / λ)
    path_loss_db = 20 * np.log10(
        4 * np.pi * distance / wavelength
    )

    # Μετατροπή από dB σε γραμμική μορφή
    path_loss = 10 ** (-path_loss_db / 10)

    # -------------------------------
    # 6. Successful transmissions counter
    # -------------------------------

    successful_transmissions = np.zeros(number_of_nodes, dtype=int)

    # -------------------------------
    # 7. Simulation over time slots
    # -------------------------------

    for slot in range(1, time_slots + 1):

        # Μία τυχαία τιμή για κάθε κόμβο
        random_values = random_generator.random(number_of_nodes)

        # Επιλογή κόμβων που εκπέμπουν
        transmit = random_values < node_probabilities

        transmitting_nodes = np.where(transmit)[0]

        # Πλήθος κόμβων που εκπέμπουν στο συγκεκριμένο slot
        N = np.sum(transmit)

        # Exponential fading
        fading = random_generator.exponential(size=N)

        # Received power at base station
        power_at_base_station = (
            fading * path_loss[transmitting_nodes]
        )

        # SINR calculation
        for i in range(N):

            node_index = transmitting_nodes[i]

            signal = power_at_base_station[i]

            # Παρεμβολή από τους υπόλοιπους κόμβους
            interference = np.sum(power_at_base_station) - signal

            # SINR = signal / (noise + interference)
            sinr = signal / (noise + interference)

            # Επιτυχής μεταφορά πακέτου
            if sinr >= gamma_threshold:
                successful_transmissions[node_index] += 1

    # -------------------------------
    # 8. Results table
    # -------------------------------

    results = pd.DataFrame({
        "node": np.arange(number_of_nodes),
        "x": x,
        "y": y,
        "distance_to_base_station": distance,
        "zone_index": node_zone_indices,
        "transmission_probability": node_probabilities,
        "path_loss_db": path_loss_db,
        "path_loss": path_loss,
        "successful_transmissions": successful_transmissions
    })

    return results


# -------------------------------
# 9. Parameters
# -------------------------------

parameters = {
    "topology": {
        "number_of_nodes": 100,
        "radius": 2000
    },

    "transmission": {
        "time_slots": 100,

        # Όρια των 20 ζωνών απόστασης
        "zone_limits": [
            100, 200, 300, 400, 500,
            600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000
        ],

        # Πιθανότητες εκπομπής ανά ζώνη
        "zone_probabilities": [
            0.70, 0.65, 0.60, 0.55, 0.50,
            0.45, 0.40, 0.35, 0.30, 0.25,
            0.20, 0.15, 0.10, 0.08, 0.06,
            0.05, 0.05, 0.05, 0.05, 0.05
        ]
    },

    "channel": {
        # Μήκος κύματος για συχνότητα 2.4 GHz
        "wavelength": 0.125,

        # Bandwidth 20 MHz
        "bandwidth": 20 * 10**6,

        # Θερμοκρασία σε Kelvin
        "temperature": 290,

        # Αρχική τιμή threshold
        "gamma_threshold": 1.0
    }
}


# -------------------------------
# 10. Run simulation for 3 thresholds
# -------------------------------

thresholds = [0.05, 0.5, 1.0]

all_results = {}

for i in range(3):

    threshold = thresholds[i]

    parameters["channel"]["gamma_threshold"] = threshold

    results_threshold = communication_simulation(parameters)

    all_results[threshold] = results_threshold

    print("\nGamma threshold:", threshold)

    print(
        "Nodes with successful packet transfer:",
        (results_threshold["successful_transmissions"] > 0).sum(),
        "out of",
        len(results_threshold)
    )

    print(
        "Total successful transmissions:",
        results_threshold["successful_transmissions"].sum()
    )

    print(
        "Proportion of nodes with successful packet transfer:",
        (results_threshold["successful_transmissions"] > 0).mean()
    )


# -------------------------------
# 11. Success Probability vs Distance
# -------------------------------

for i in range(3):

    threshold = thresholds[i]

    results_threshold = all_results[threshold]

    success_probability = (
        results_threshold["successful_transmissions"]
        / parameters["transmission"]["time_slots"]
    )

    plt.plot(
        results_threshold["distance_to_base_station"],
        success_probability,
        "o"
    )

plt.xlabel("Distance from base station (m)")
plt.ylabel("Success probability per slot")
plt.title("Success Probability vs Distance")

plt.xlim(0, parameters["topology"]["radius"])
plt.ylim(0, 1)

plt.grid(True)
plt.show()


# -------------------------------
# 12. Successful Transmissions vs Distance
# -------------------------------

for i in range(3):

    threshold = thresholds[i]

    results_threshold = all_results[threshold]

    plt.plot(
        results_threshold["distance_to_base_station"],
        results_threshold["successful_transmissions"],
        "o"
    )

plt.xlabel("Distance from base station (m)")
plt.ylabel("Successful transmissions")
plt.title("Successful Transmissions vs Distance")

plt.xlim(0, parameters["topology"]["radius"])
plt.ylim(0, parameters["transmission"]["time_slots"])

plt.grid(True)
plt.show()
