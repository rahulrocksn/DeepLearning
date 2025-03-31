#
import argparse
import os
import sys
import torch
import numpy as onp
import more_itertools as xitertools
import hashlib
import random
from typing import Dict
from structures.ddi import MetaDDI
from models.gcn import GCN, ModelGCN
from utils import sgen, negative_sampling, edge_index_to_vector

from tqdm import tqdm


# Set quick flushing for slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)


def describe(hash: bool, /, *, random_mask: bool, positional: bool) -> str:
    R"""
    Get description.
    """
    #
    description = "ddi"
    if random_mask:
        description += "_random-mask"
    if positional:
        #
        description += "_position"
    else:
        #
        description += "_structure"

    #
    if hash:
        #
        description = hashlib.md5(description.encode()).hexdigest()
    return description


class Evaluator():
    R"""
    Hits-before-K Evaluator.
    """
    #
    K = 20

    #
    @classmethod
    def eval(cls, scores: Dict[str, torch.Tensor], /) -> Dict[str, float]:
        R"""
        Evaluate.
        """
        #
        y_pred_pos = scores["y_pred_pos"]
        y_pred_neg = scores["y_pred_neg"]

        #
        if len(y_pred_neg) < cls.K:
            #
            return {"hits@{:d}".format(cls.K): 1.0}
        else:
            #
            kth_score_in_negative_edges = torch.topk(y_pred_neg, cls.K)[0][-1]
            hitsK = (
                float(
                    torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu(),
                )
                / float(len(y_pred_pos))
            )
            return {"hits@{:d}".format(cls.K): hitsK}


@torch.no_grad()
def evaluate(
    metaset: MetaDDI, model: ModelGCN, evaluator: Evaluator, indices: str,
    /,
    *,
    device: str, batch_size: int,
) -> float:
    R"""
    Evaluate.
    """
    # Formalize metaset.
    (edges_observe, _, edges_valid, edges_test) = metaset
    edges_dict = dict(valid=edges_valid, test=edges_test)
    (edges_pos, edges_neg) = edges_dict[indices]
    edges_eval = dict(pos=edges_pos.to(device), neg=edges_neg.to(device))
    edges_score = {}

    # Node embeddings are shared and should be computed only once.
    model.eval()
    (edge_indices, edge_weights) = edges_observe
    node_embeds = model.encode(edge_indices, edge_weights)
    for label in ("pos", "neg"):
        #
        buf = []
        for batch_indices in (
            xitertools.sliced(range(edges_eval[label].shape[-1]), batch_size)
        ):
            #
            edge_probs = (
                model.decode(
                    node_embeds, edges_eval[label][:, list(batch_indices)],
                )
            )
            buf.append(edge_probs.squeeze().cpu())
        edges_score[label] = torch.cat(buf)

    #
    hits20: float

    # Get results.
    hits20 = (
        evaluator.eval(
            {
                "y_pred_pos": edges_score["pos"],
                "y_pred_neg": edges_score["neg"],
            },
        )["hits@20"]
    )
    return hits20


def train(
    metaset: MetaDDI, model: ModelGCN, optimizer: torch.optim.Optimizer,
    indices: str, random_mask: bool,
    /,
    *,
    device: str, batch_size: int, rng: onp.random.RandomState, seed: int,
) -> None:
    R"""
    Train.
    """
    # Formalize metaset.
    (edges_observe, edges_pos_train, _, _) = metaset
    edges_pos_train = edges_pos_train.to(device)

    #
    model.train()
    (edge_indices, edge_weights) = edges_observe

    # If random masking, discard the predefined edges_pos_train.
    # Use the same ratio to sample new edges_pos_train
    if random_mask:
        ratio = edges_pos_train.shape[1] / edge_indices.shape[1]
        n_pos = int(ratio * edge_indices.shape[1])
        edges_pos_train_ids = rng.choice(edge_indices.shape[1], n_pos, replace=False)
        edges_pos_train = edge_indices[:, edges_pos_train_ids]

    #
    full_indices = onp.arange(edges_pos_train.shape[-1])
    full_perm = rng.permutation(len(full_indices)) if random_mask else onp.arange(len(full_indices))
    random.seed(seed)
    for batch_indices in (
        xitertools.sliced(full_indices[full_perm].tolist(), batch_size)
    ):
        #
        edges_pos = edges_pos_train[:, batch_indices]
        edges_neg = (
            negative_sampling(
                edge_indices, model.num_nodes, len(batch_indices),
            )
        )

        #
        optimizer.zero_grad()
        node_embeds = model.encode(edge_indices, edge_weights)
        edge_probs_pos = model.decode(node_embeds, edges_pos)
        edge_probs_neg = model.decode(node_embeds, edges_neg)

        #
        loss_pos = -torch.log(edge_probs_pos + 1e-15).mean()
        loss_neg = -torch.log(1 - edge_probs_neg + 1e-15).mean()
        loss = loss_pos + loss_neg
        loss.backward()

        #
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def main(*ARGS) -> None:
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Main Execution (Homework 2)")
    parser.add_argument(
        "--source",
        type=str, required=False, default="data",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--random-seed",
        type=int, required=False, default=47, help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str, required=False, default="cpu", help="Device.",
    )
    parser.add_argument(
        "--random-mask",
        action="store_true",
        help="Use random masking during training instead of predefined.",
    )
    parser.add_argument(
        "--positional",
        action="store_true",
        help="Use positional embedding instead of structural.",
    )
    parser.add_argument(
        "--epochs",
        type=int, required=False, default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float, required=False, default=0.005, help="Learning rate.",
    )
    parser.add_argument(
        "--batch",
        type=int, required=False, default=64 * 1024, help="Batch size.",
    )
    parser.add_argument(
        "--hidden",
        type=int, required=False, default=256,
        help="Size of hidden embedding.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    source = args.source
    seed = args.random_seed
    device = args.device
    positional = args.positional

    #
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    hidden = args.hidden

    #
    description = describe(False, random_mask=args.random_mask, positional=positional)
    print("\x1b[104;30m{:s}\x1b[0m".format(description))

    # We only have a single undirected graph in DDI.
    # Observed edges should be directly pinned into computation memory.
    with open(os.path.join(source, "ddi.npy"), "rb") as file:
        #
        array_num_nodes = onp.load(file)
        array_edge_indices = onp.load(file)
        array_edge_pos_train = onp.load(file)
        array_edge_pos_valid = onp.load(file)
        array_edge_neg_valid = onp.load(file)
        array_edge_pos_test = onp.load(file)
        array_edge_neg_test = onp.load(file)
    num_nodes = array_num_nodes.item()
    edge_indices = torch.from_numpy(array_edge_indices)
    edge_pos_train = torch.from_numpy(array_edge_pos_train)
    edge_pos_valid = torch.from_numpy(array_edge_pos_valid)
    edge_neg_valid = torch.from_numpy(array_edge_neg_valid)
    edge_pos_test = torch.from_numpy(array_edge_pos_test)
    edge_neg_test = torch.from_numpy(array_edge_neg_test)
    edge_weights = GCN.degree_normalizor(num_nodes, edge_indices)

    # Safety check
    edges_train1 = edge_pos_train.T
    edges_train2 = edge_indices[:, ::2]
    if (
        edges_train1.shape != edges_train2.shape
        or torch.any(edges_train1 != edges_train2).item()
    ):
        # UNEXPECT:
        # Defined and defaul training edges conflict.
        raise NotImplementedError(
            "Defined and default training edges conflict.",
        )

    #
    edge_indices = edge_indices.to(device)
    edge_weights = edge_weights.to(device)

    #
    metaset = (
        (edge_indices, edge_weights), edge_pos_train.T,
        (edge_pos_valid.T, edge_neg_valid.T),
        (edge_pos_test.T, edge_neg_test.T),
    )

    #
    for directory in ("ptlog", "ptnnp"):
        #
        if not os.path.isdir(directory):
            #
            os.makedirs(directory)
    path_log = os.path.join("ptlog", "{:s}.ptlog".format(description))
    path_nnp = os.path.join("ptnnp", "{:s}.ptnnp".format(description))

    #
    model = ModelGCN(hidden, num_nodes, 2, positional=positional)
    thrng = torch.Generator("cpu")
    thrng.manual_seed(seed)
    model.initialize(thrng)
    model = model.to(device)

    #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    evaluator = Evaluator()
    evaluator.K = 20
    log = []

    #
    print("{:s} {:s} {:s}".format("-" * 5, "-" * 13, "-" * 7))
    print("{:>5s} {:>13s} {:>4s}".format("Epoch", "Valid Hits@20", "Best"))
    print("{:s} {:s} {:s}".format("-" * 5, "-" * 13, "-" * 7))
    hits20 = (
        evaluate(
            metaset, model, evaluator, "valid",
            device=device, batch_size=batch_size,
        )
    )
    best_hits20 = hits20
    torch.save(model.state_dict(), path_nnp)
    log.append(hits20)
    torch.save(log, path_log)
    print(
        "{:>5d} {:>13s} {:.5f}"
        .format(0, "{:.5f}".format(hits20), best_hits20),
    )
    nprng = onp.random.RandomState(seed)
    for epoch in range(1, epochs + 1):
        #
        train(
            metaset, model, optimizer, "valid", args.random_mask,
            device=device, batch_size=batch_size, rng=nprng, seed=seed + epoch,
        )
        if epoch % 5 > 0:
            #
            continue
        hits20 = (
            evaluate(
                metaset, model, evaluator, "valid",
                device=device, batch_size=batch_size,
            )
        )
        if best_hits20 < hits20:
            #
            best_hits20 = hits20
            torch.save(model.state_dict(), path_nnp)
        log.append(hits20)
        torch.save(log, path_log)
        print(
            "{:>5d} {:>13s} {:.5f}"
            .format(epoch, "{:.5f}".format(hits20), best_hits20),
        )
    print("{:s} {:s} {:s}".format("-" * 5, "-" * 13, "-" * 7))

    #
    model.load_state_dict(torch.load(path_nnp, map_location=device))
    hits20 = (
        evaluate(
            metaset, model, evaluator, "test",
            device=device, batch_size=batch_size,
        )
    )
    print("- Test Hits@20: {:.5f}".format(hits20))


if __name__ == "__main__":
    #
    main()