import os
import sys

import torch.nn as nn
import torch
import pandas as pd
import numpy as np

# Find current device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Only import cupy in CUDA environment.
if device == "cuda":
    import cupy as cp

# ---------------------------- Dataset Processing -------------------------


def load_dataset(input_dir, num, pt_background_cut, pt_signal_cut, true_edges, noise):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event)
                            for event in all_events])

#        loaded_events = [
#            torch.load(event, map_location=torch.device("cpu"))
#            for event in all_events[:num]
#        ]

        loaded_events = []
        for event in all_events[:num]:
            try:
                loaded_events.append(torch.load(
                    event, map_location=torch.device("cpu")))
            except Exception as e:
                print(f"{e} occurs while processing event {event}")

        loaded_events = select_data(
            loaded_events, pt_background_cut, pt_signal_cut, true_edges, noise)
        return loaded_events
    else:
        return None


def get_edge_subset(edges, mask_where, inverse_mask):

    included_edges_mask = np.isin(edges, mask_where).all(0)
    included_edges = edges[:, included_edges_mask]
    included_edges = inverse_mask[included_edges]

    return included_edges, included_edges_mask


def select_data(events, pt_background_cut, pt_signal_cut, true_edges, noise):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if (pt_background_cut > 0) or not noise:
        for event in events:

            pt_mask = (event.pt > pt_background_cut) & (event.pid == event.pid)
            pt_where = torch.where(pt_mask)[0]

            inverse_mask = torch.zeros(pt_where.max()+1).long()
            inverse_mask[pt_where] = torch.arange(len(pt_where))

            edge_mask = None
            event[true_edges], edge_mask = get_edge_subset(
                event[true_edges], pt_where, inverse_mask)

            if "weights" in event.__dict__.keys():
                if event.weights.shape[0] == event[true_edges].shape[1]:
                    event.weights = event.weights[edge_mask]

            event.edge_index, _ = get_edge_subset(
                event.edge_index, pt_where, inverse_mask)

            node_features = ["cell_data", "x", "hid", "pid", "pt", "layers"]
            for feature in node_features:
                if feature in event.__dict__.keys():
                    event[feature] = event[feature][pt_mask]

    # Define the signal edges
    for event in events:
        if pt_signal_cut > 0:
            edge_subset = (event.pt[event[true_edges]] > pt_signal_cut).all(0)
            event.signal_true_edges = event[true_edges][:, edge_subset]
        else:
            event.signal_true_edges = event[true_edges]

    return events


def random_edge_slice_v2(delta_phi, batch):
    """
    Same behaviour as v1, but avoids the expensive calls to np.isin and np.unique, using sparse operations on GPU
    """
    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu").numpy()
    x = batch.x.to("cpu")

    # 2. Find edges within delta_phi of random_phi
    e_average = (x[e[0], 1] + x[e[1], 1]) / 2
    dif = abs(e_average - random_phi)
    subset_edges = ((dif < delta_phi) | ((2 - dif) < delta_phi)).numpy()

    # 3. Find connected edges to this subset
    e_ones = cp.array([1] * e_length).astype("Float32")
    subset_ones = cp.array([1] * subset_edges.sum()).astype("Float32")

    e_csr_in = cp.sparse.coo_matrix(
        (
            e_ones,
            (cp.array(e[0]).astype("Float32"),
             cp.arange(e_length).astype("Float32")),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    e_csr_out = cp.sparse.coo_matrix(
        (
            e_ones,
            (cp.array(e[0]).astype("Float32"),
             cp.arange(e_length).astype("Float32")),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    e_csr = e_csr_in + e_csr_out

    subset_csr_in = cp.sparse.coo_matrix(
        (
            subset_ones,
            (
                cp.array(e[0, subset_edges]).astype("Float32"),
                cp.arange(e_length)[subset_edges].astype("Float32"),
            ),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    subset_csr_out = cp.sparse.coo_matrix(
        (
            subset_ones,
            (
                cp.array(e[0, subset_edges]).astype("Float32"),
                cp.arange(e_length)[subset_edges].astype("Float32"),
            ),
        ),
        shape=(e.max() + 1, e_length),
    ).tocsr()
    subset_csr = subset_csr_in + subset_csr_out

    summed = (subset_csr.T * e_csr).sum(axis=0)
    subset_edges_extended = (summed > 0)[0].get()

    return subset_edges, subset_edges_extended


def random_edge_slice(delta_phi, batch):
    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    # 2. Find hits within delta_phi of random_phi
    dif = abs(x[:, 1] - random_phi)
    subset_hits = np.where((dif < delta_phi) | ((2 - dif) < delta_phi))[0]

    # 3. Filter edges with subset_hits
    subset_edges_ind = np.isin(e[0], subset_hits) | np.isin(e[1], subset_hits)

    subset_hits = np.unique(e[:, subset_edges_ind])
    subset_edges_extended = np.isin(
        e[0], subset_hits) | np.isin(e[1], subset_hits)
    nested_ind = np.isin(
        np.where(subset_edges_extended)[0], np.where(subset_edges_ind)[0]
    )

    return subset_edges_ind, subset_edges_extended, nested_ind


def hard_random_edge_slice(delta_phi, batch):

    # 1. Select random phi
    random_phi = np.random.rand() * 2 - 1
    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    e_average = (x[e[0], 1] + x[e[1], 1]) / 2
    dif = abs(e_average - random_phi)
    subset_edges_ind = ((dif < delta_phi) | ((2 - dif) < delta_phi)).numpy()

    return subset_edges_ind


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1.0 * np.log(np.tan(theta / 2.0))


def hard_eta_edge_slice(delta_eta, batch):

    e = batch.e_radius.to("cpu")
    x = batch.x.to("cpu")

    etas = calc_eta(x[:, 0], x[:, 2])
    random_eta = (np.random.rand() - 0.5) * 2 * (etas.max() - delta_eta)

    e_average = (etas[e[0]] + etas[e[1]]) / 2
    dif = abs(e_average - random_eta)
    subset_edges_ind = ((dif < delta_eta)).numpy()

    return subset_edges_ind


# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


# ----------------------------- Performance Utilities ---------------------------


def graph_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.1,
        x1=0.2,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(edge_cut, model, trainer, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(edge_cut, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results):

    ps = [result["preds"].sum() for result in test_results[1:]]
    ts = [result["truth"].sum() for result in test_results[1:]]
    tps = [(result["preds"] * result["truth"]).sum()
           for result in test_results[1:]]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity


def evaluate_set_metrics(edge_cut, model, trainer):
    model.hparams.edge_cut = edge_cut
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity = get_metrics(test_results)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity
