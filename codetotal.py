import numpy as np
import pandas as pd


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

    # Θερμικός θόρυβος:
    # N = k * T * B
    boltzmann_constant = 1.38 * 10**(-23)
    noise = boltzmann_constant * temperature * bandwidth

    # -------------------------------
    # 2. Random generator
    # -------------------------------

    random_generator = np.random.default_rng()

    # -------------------------------
    # 3. Random node distances
    # -------------------------------

    distance = radius * random_generator.random(number_of_nodes)

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
    # 6. State variable
    # -------------------------------

    # Μας ενδιαφέρει ποιοι κόμβοι πέτυχαν μεταφορά πακέτου.
    packet_transfer_success = np.zeros(number_of_nodes, dtype=bool)

    # -------------------------------
    # 7. Simulation over time slots
    # -------------------------------

    for slot in range(1, time_slots + 1):

        # Μία τυχαία τιμή για κάθε κόμβο.
        random_values = random_generator.random(number_of_nodes)

        # Αν random_value < πιθανότητα εκπομπής,
        # τότε ο κόμβος εκπέμπει.
        transmit = random_values < node_probabilities

        # Βρίσκουμε τους κόμβους που έχουν True,
        # δηλαδή αυτούς που εκπέμπουν στο συγκεκριμένο slot.
        transmitting_nodes = np.where(transmit)[0]

        # -------------------------------
        # Fading
        # -------------------------------

        # Για αρχή παίρνουμε h ως τυχαία τιμή από 0 έως 1.
        h = random_generator.random(len(transmitting_nodes))

        # Χρησιμοποιούμε h^2 ως fading στην ισχύ.
        fading = h**2

        # Ισχύς που φτάνει στο base station από κάθε κόμβο που εκπέμπει.
        # P_r = h^2 * path_loss

        power_at_base_station = (
            fading * path_loss[transmitting_nodes]
        )

        # -------------------------------
        # SINR calculation
        # -------------------------------

        for i in range(len(transmitting_nodes)):

            node_index = transmitting_nodes[i]

            signal = power_at_base_station[i]

            # Παρεμβολή = άθροισμα των σημάτων των άλλων κόμβων
            # που εκπέμπουν στο ίδιο slot.

            interference = np.sum(power_at_base_station) - signal

            # SINR = signal / (noise + interference)

            sinr = signal / (noise + interference)

            # Επιτυχής μεταφορά πακέτου αν SINR >= gamma_threshold

            if sinr >= gamma_threshold:
                packet_transfer_success[node_index] = True

    # -------------------------------
    # 8. Results table
    # -------------------------------

    results = pd.DataFrame({
        "node": np.arange(number_of_nodes),
        "distance_to_base_station": distance,
        "zone_index": node_zone_indices,
        "transmission_probability": node_probabilities,
        "path_loss_db": path_loss_db,
        "path_loss": path_loss,
        "packet_transfer_success": packet_transfer_success
    })

    return results


# -------------------------------
# Parameters
# -------------------------------

parameters = {
    "topology": {
        "number_of_nodes": 100,
        "radius": 2000
    },

    "transmission": {
        "time_slots": 100,

        # Όρια ζωνών απόστασης.
        # Το zone_limits είναι απλός πίνακας τιμών.
        # Η np.digitize χρησιμοποιεί αυτές τις τιμές ως όρια.

        "zone_limits": [
            100, 200, 300, 400, 500,
            600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000
        ],

        # Πιθανότητες εκπομπής ανά ζώνη.
        # Η θέση 0 αντιστοιχεί στη ζώνη 0,
        # η θέση 1 στη ζώνη 1, κ.ο.κ.

        "zone_probabilities": [
            0.70, 0.65, 0.60, 0.55, 0.50,
            0.45, 0.40, 0.35, 0.30, 0.25,
            0.20, 0.15, 0.10, 0.08, 0.06,
            0.05, 0.05, 0.05, 0.05, 0.05
        ]
    },

    "channel": {
        # Για συχνότητα f = 2.4 GHz:
        # λ = c / f
        # λ = (3 * 10**8) / (2.4 * 10**9)
        # λ = 0.125 m

        "wavelength": 0.125,

        # Bandwidth καναλιού.
        # 20 * 10**6 σημαίνει 20 MHz.

        "bandwidth": 20 * 10**6,

        # Θερμοκρασία σε Kelvin.

        "temperature": 290,

        # SINR threshold.
        # Η μεταφορά πακέτου πετυχαίνει αν SINR >= gamma_threshold.

        "gamma_threshold": 1.0
    }
}


# -------------------------------
# Run simulation
# -------------------------------

results = communication_simulation(parameters)

print(results)

print(
    "\nNodes with successful packet transfer:",
    results["packet_transfer_success"].sum(),
    "out of",
    len(results)
)

print(
    "Packet transfer success rate:",
    results["packet_transfer_success"].mean()
)