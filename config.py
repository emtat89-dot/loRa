PARAMETERS = {
    "topology": {
        "number_of_nodes": 100,
        "radius": 2000
    },
    "transmission": {
        "time_slots": 100,
        "zone_limits": [
            100, 200, 300, 400, 500,
            600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500,
            1600, 1700, 1800, 1900, 2000
        ],
        "zone_probabilities": [
            0.70, 0.65, 0.60, 0.55, 0.50,
            0.45, 0.40, 0.35, 0.30, 0.25,
            0.20, 0.15, 0.10, 0.08, 0.06,
            0.05, 0.05, 0.05, 0.05, 0.05
        ]
    },
    "channel": {
        "wavelength": 0.125,
        "bandwidth": 20e6,
        "temperature": 290,
        "gamma_threshold": 1.0
    }
}
