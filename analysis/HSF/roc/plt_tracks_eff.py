import sys
import os
import logging
logging.basicConfig(level=print, format='%(levelname)s:%(message)s')

import torch
from torch_geometric.data import Data

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sps

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial

from utils.convenience_utils import headline, delete_directory
from utils.plotting_utils import plot_pt_eff

matplotlib.use('Agg')
sys.path.append("../../")

config = {

    "input_dir": "/global/cfs/cdirs/m3443/usr/daniel/dataset/gnn/",

    # track building configs
    "score_cut": 0.9,

    # evaluation configs
    "min_pt": 1,
    "max_eta": 4,
    "min_track_length": 3,
    "min_particle_length": 3,
    "matching_fraction": 0.5,
    "matching_style": "ATLAS"

}

def label_graph(graph, score_cut=0.8):

    """
    Find the connected edges (tracks) from the given graph (in matrix form). Return the number of connected components 
    and a length-N array of labels
    
    Return example: 
    5, (0, 1, 1, 0, 2, 2, 1)
    """

    # os.makedirs(save_dir, exist_ok=True)

    edge_mask = graph.score > score_cut

    row, col = graph.edge_index[:, edge_mask]
    edge_attr = np.ones(row.size(0))

    N = graph.x.size(0) # number of hits
    sparse_edges = sps.coo_matrix((edge_attr, (row.numpy(), col.numpy())), (N, N))

    _, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)  
    graph.labels = torch.from_numpy(candidate_labels).long()

    return graph


def get_labelled_graphs():

    """
    Give the track label to each graph in input_dir and output the graph data with label attribute
    """

    print(headline( " Building track candidates from the scored graph " ))
    print(headline("a) Loading scored graphs" ))

    all_graphs = []
    # should probably only use "test"
    for subdir in ["train", "val", "test"]:
        subdir_graphs = os.listdir(os.path.join(config["input_dir"], subdir))
        all_graphs += [torch.load(os.path.join(config["input_dir"], subdir, graph), map_location="cpu") for graph in subdir_graphs]

    print(headline( "b) Labelling graph nodes" ) )

    score_cut = config["score_cut"]
    
    # RUN IN SERIAL FOR NOW -->
    labelled_graphs = []
    for graph in tqdm(all_graphs):
        graph.event_id = int(graph.event_file[-4:])
        labelled_graphs.append(label_graph(graph, score_cut=score_cut))
    
    return labelled_graphs

def load_reconstruction_df(graph: Data):
    """Load the reconstructed tracks from a file."""
    graph = graph.to(device='cpu')

    # track_id is the reconstructed track, pid is the real track
    reconstruction_df = pd.DataFrame({"hit_id": graph.hid, "track_id": graph.labels, "particle_id": graph.pid})
    return reconstruction_df

def load_particles_df(graph: Data):
    """Load the particles from a file."""
    graph = graph.to('cpu')

    # Get the particle dataframe
    particles_df = pd.DataFrame({"particle_id": graph.pid})
    # particles_df = pd.DataFrame({"particle_id": graph.pid, "pt": graph.pt, "eta": graph.eta})

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=['particle_id'])

    return particles_df

def get_matching_df(reconstruction_df, particles_df, min_track_length=1, min_particle_length=1):
    
    # Get number of hits in each track => dataframe with col[track_id, num_hits]
    candidate_lengths = reconstruction_df.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # Get number of hits in each real track => dataframe with col[pid, num_hits]
    particle_lengths = reconstruction_df.drop_duplicates(subset=['hit_id']).particle_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"particle_id", "particle_id": "n_true_hits"})

    spacepoint_matching = reconstruction_df.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})
    

    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=['track_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particle_lengths, on=['particle_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particles_df, on=['particle_id'], how='left')

    """
    Get a dataFrame such with the form
    track_id  pid  shared  n_reco_hits  n_true_hits  pt    eta
    1         1    2       5            3            ...   ...  => 5 spacepoints with track_id = 1, 3 with pid = 1, 2 with both
    1         2    4       5            7            ...   ...  => 5 spacepoints with track_id = 1, 7 with pid = 2, 4 with both
    ...

    98        72   8       15           17           ...   ...  => 15 spacepoints with track_id = 98, 17 with pid = 72, 8 with both
    
    Note:
        -> shared / candidate_lengths: how many nodes in the reconstructed track are actually belong to same real track -> purity
        -> shared / particle_lengths: how many nodes in the real track are actually reconstructed -> efficiency
    """

    # Filter out tracks with too few shared spacepoints
    spacepoint_matching["is_matchable"] = spacepoint_matching.n_reco_hits >= min_track_length
    spacepoint_matching["is_reconstructable"] = spacepoint_matching.n_true_hits >= min_particle_length

    return spacepoint_matching

def calculate_matching_fraction(spacepoint_matching_df):
    # get the purity of each row of data by performing n_shared / n_reco_hits for each row.
    # Name the output attribute as "purity_reco"
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits))
    
    # get the efficiency of each row of data by performing n_shared / n_true_hits for each row.
    # Name the output attribute as "eff_true"
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true = np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits))

    return spacepoint_matching_df

def evaluate_labelled_graph(graph, matching_fraction=0.5, matching_style="ATLAS", min_track_length=1, min_particle_length=1):

    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be >= 0.5")

    if matching_fraction == 0.5:
        # Add a tiny bit of noise to the matching fraction to avoid double-matched tracks
        matching_fraction += 0.00001

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph)
    particles_df = load_particles_df(graph)

    # Get matching dataframe
    matching_df = get_matching_df(reconstruction_df, particles_df, min_track_length=min_track_length, min_particle_length=min_particle_length) 
    matching_df["event_id"] = graph.event_id

    # calculate matching fraction
    matching_df = calculate_matching_fraction(matching_df)

    # Run matching depending on the matching style
    if matching_style == "ATLAS":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = matching_df.purity_reco >= matching_fraction
    elif matching_style == "one_way":
        matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
        matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
    elif matching_style == "two_way":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (matching_df.purity_reco >= matching_fraction) & (matching_df.eff_true >= matching_fraction)

    return matching_df


def evaluate(graphs):

    print(headline("Evaluating the track reconstruction performance"))
    print(headline("a) Loading labelled graphs"))

    evaluated_events = []
    for graph in tqdm(graphs):
        evaluated_events.append(evaluate_labelled_graph(graph, 
                                matching_fraction=config["matching_fraction"], 
                                matching_style=config["matching_style"],
                                min_track_length=config["min_track_length"],
                                min_particle_length=config["min_particle_length"]))
    evaluated_events = pd.concat(evaluated_events)

    """
    1. For recontruction efficiency
        1) find the recontructable particles in events file (no drop duplicated yet)
        2) find the particles that are acttually recontructed (no drop duplicated yet)
        3) drop the duplicated particles in above examples and find the numbers of all particles and recontructed particles
        4) calculate recontruction effiiciency
    
    2. For matching puritty
        1) find the matchable track in events file (no drop duplicated yet)
        2) find the tracks actually match a particles (no drop duplicated track yet)
        3) drop the duplicated tracks in above exmples and find the number of all tracks and all matched tracks
        4) calculate matcing purity

    """

    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[particles["is_reconstructed"] & particles["is_matchable"]]    
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=['event_id', 'particle_id']))
    n_reconstructed_particles = len(reconstructed_particles.drop_duplicates(subset=['event_id', 'particle_id']))
    
    n_tracks = len(tracks.drop_duplicates(subset=['event_id', 'track_id']))
    n_matched_tracks = len(matched_tracks.drop_duplicates(subset=['event_id', 'track_id']))

    n_dup_reconstructed_particles = len(reconstructed_particles) - n_reconstructed_particles

    print(headline("b) Calculating the performance metrics"))
    print(f"Number of reconstructed particles: {n_reconstructed_particles}")
    print(f"Number of particles: {n_particles}")
    print(f"Number of matched tracks: {n_matched_tracks}")
    print(f"Number of tracks: {n_tracks}")
    print(f"Number of duplicate reconstructed particles: {n_dup_reconstructed_particles}")   

    # Plot the results across pT and eta
    eff = n_reconstructed_particles / n_particles
    fake_rate = 1 - (n_matched_tracks / n_tracks)
    dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles
    
    print(f"Efficiency: {eff:.3f}")
    print(f"Fake rate: {fake_rate:.3f}")
    print(f"Duplication rate: {dup_rate:.3f}")

    print(headline("c) Plotting results"))

    # First get the list of particles without duplicates
    grouped_reco_particles = particles.groupby('particle_id')["is_reconstructed"].any()
    particles["is_reconstructed"] = particles["particle_id"].isin(grouped_reco_particles[grouped_reco_particles].index.values)
    particles = particles.drop_duplicates(subset=['particle_id'])

    # Plot the results across pT and eta
    plot_pt_eff(particles)

    # save file
    """
    'track_id', 'particle_id', 'n_shared', 'n_reco_hits', 'n_true_hits',
       'pt', 'is_matchable', 'is_reconstructable', 'event_id', 'purity_reco',
       'eff_true', 'is_matched', 'is_reconstructed'
    """

    return evaluated_events, reconstructed_particles, particles, matched_tracks, tracks
    



if __name__ == "__main__":

    graphs = get_labelled_graphs() 
    evaluate(graphs) 
