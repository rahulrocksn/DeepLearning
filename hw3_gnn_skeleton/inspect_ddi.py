"""
    Inspect the Drug-Drug Interaction dataset statistics
"""
import os
import argparse

import numpy as onp
import torch
import matplotlib.pyplot as plt



def compute_degrees(num_nodes, edge_indices):
    """
    Compute the degree of each node in the graph, given the edge indices.
    """
    # YOU NEED TO FILL IN THIS PART.
    # Hint: Check out PyTorch's `torch.zeros` and `scatter_add_` functions. They might be useful.
    # Also be ware of the shape of the edge_indices tensor.
    ...


def concatenate_edges(edge_indices_1, edge_indices_2):
    """
    Concatenate two edge index tensors.
    """
    # YOU NEED TO FILL IN THIS PART.
    # Hint: Check out PyTorch's `torch.cat` function.
    # Be ware of the shape of the edge_indices tensors. If you call `torch.cat`, what is the correct dimension to concatenate on?
    ...


def plot_degree_histogram(degrees, plot_dir):
    """
    Plot 3 degree histogram subfigures in a row.
    First plot have both x and y axis in linear scale
    Second plot have y axis in log scale
    Third plot have both x and y axis in log scale
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(degrees, bins=50, color="blue", alpha=0.7)
    axs[0].set_xlabel("Degree")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Degree Histogram (Linear Scale)")

    axs[1].hist(degrees, bins=50, color="blue", alpha=0.7)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Degree")
    axs[1].set_ylabel("Count (log scale)")
    axs[1].set_title("Degree Histogram (Log Scale)")

    axs[2].hist(degrees, bins=50, color="blue", alpha=0.7)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Degree (log scale)")
    axs[2].set_ylabel("Count (log scale)")
    axs[2].set_title("Degree Histogram (Log-Log Scale)")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "degree_histogram.pdf"))
    plt.close()


def print_degree_statistics(train_degrees, valid_degrees, test_degrees):
    """
    Print the statistics of the training, validation, and test degrees.
    """
    print("Degree Statistics:")
    print(f"\tMin\tMax\tAverage")
    print(f"Train\t{train_degrees.min().item()}\t{train_degrees.max().item()}\t{train_degrees.float().mean().item():.3f}")
    print(f"Valid\t{valid_degrees.min().item()}\t{valid_degrees.max().item()}\t{valid_degrees.float().mean().item():.3f}")
    print(f"Test\t{test_degrees.min().item()}\t{test_degrees.max().item()}\t{test_degrees.float().mean().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect DDI")

    parser.add_argument(
        "--source",
        type=str, required=False, default="data",
        help="Source root directory for data.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str, required=False, default="plots",
        help="Directory to save plots.",
    )

    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    # Load DDI data
    ddi_path = os.path.join(args.source, "ddi.npy")
    if not os.path.isfile(ddi_path):
        raise ValueError("DDI data not found. Please download it first by running `python download.py`")

    with open(os.path.join(args.source, "ddi.npy"), "rb") as file:
        array_num_nodes = onp.load(file)
        array_edge_indices_all = onp.load(file)
        array_edge_pos_train = onp.load(file)
        array_edge_pos_valid = onp.load(file)
        array_edge_neg_valid = onp.load(file)
        array_edge_pos_test = onp.load(file)
        array_edge_neg_test = onp.load(file)
    num_nodes = array_num_nodes.item()
    edge_indices_all = torch.from_numpy(array_edge_indices_all)
    edge_pos_train = torch.from_numpy(array_edge_pos_train).T   # Transpose to make it same dimension as edge_indices
    edge_pos_valid = torch.from_numpy(array_edge_pos_valid).T
    edge_pos_test = torch.from_numpy(array_edge_pos_test).T

    # YOU NEED TO FILL IN THIS PART.
    # Your TODOs: 
    # - get the undirected edge indices for the entire graph, the train, valid, and test splits by concatenating the original edge indices with their reverse.
    # - compute the degrees of these edge indices by calling the `compute_degrees` function.
    # - plot the degree histogram of the entire graph by calling the `plot_degree_histogram` function.
    # - print the degree statistics of the training, validation, and test splits by calling the `print_degree_statistics` function.
    edge_indices_all = concatenate_edges(edge_indices_all, edge_indices_all.flip(0))
    edge_pos_train = concatenate_edges(edge_pos_train, edge_pos_train.flip(0))
    edge_pos_valid = concatenate_edges(edge_pos_valid, edge_pos_valid.flip(0))
    edge_pos_test = concatenate_edges(edge_pos_test, edge_pos_test.flip(0))

    degrees_all = compute_degrees(num_nodes, edge_indices_all)
    degrees_pos_train = compute_degrees(num_nodes, edge_pos_train)
    degrees_pos_valid = compute_degrees(num_nodes, edge_pos_valid)
    degrees_pos_test = compute_degrees(num_nodes, edge_pos_test)

    plot_degree_histogram(degrees_all, args.plot_dir)
    
    print_degree_statistics(degrees_pos_train, degrees_pos_valid, degrees_pos_test)


if __name__ == "__main__":
    main()
