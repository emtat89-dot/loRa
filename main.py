import argparse

import pandas as pd

from config import PARAMETERS
from simulation import communication_simulation


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    parameters = PARAMETERS
    if arguments.nodes is not None:
        parameters["topology"]["number_of_nodes"] = arguments.nodes
    if arguments.slots is not None:
        parameters["transmission"]["time_slots"] = arguments.slots

    results = communication_simulation(parameters, seed=arguments.seed)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(results)

    print("\nNodes with at least one successful transmission:",
          results["success"].sum(), "out of", len(results))
    print("Success rate:", results["success"].mean())
    print("Total transmissions:", results["attempts"].sum())
    print("Total successful transmissions:",
          results["successful_transmissions"].sum())

    if arguments.output:
        results.to_csv(arguments.output, index=False)
        print("Results saved to", arguments.output)


if __name__ == "__main__":
    main()
