#!/usr/bin/env python3

from pathlib import Path
import itertools
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

"""
Compute and plot filter score distribution.

Filter stage don't record score by default.
We need to recompute it instead of getting it from log. 
"""


def load_model(ckpt, device=None):
    # We can symlink model to same folder for convenience
    from Filter.Models.vanilla_filter import VanillaFilter
    from Filter.Models.pyramid_filter import PyramidFilter
    
    device = device or 'cuda'
    
    e_ckpt = torch.load(ckpt, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_config['input_dir'] = "/global/cfs/cdirs/m3443/data/LRTTraining/v5/embedding_processed_0.3_900"
    e_model = PyramidFilter(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.to(device)
    e_model.eval()
    e_model.setup('test')

    return e_model


def test_model(model, device=None):
    """
    Test model.

    @return pur, eff
    """
    device = device or 'cuda'
    
    # Initialize result statistics.
    result = {
        'truth': [],
        'score': []
    }
    
    for batch_idx, batch in enumerate(tqdm(model.test_dataloader())):
        with torch.no_grad():
            batch = batch.to(device)
            eval_result = model.shared_evaluation(
                batch, batch_idx
            )

            result['truth'].append(eval_result['truth'].detach().cpu().numpy())
            result['score'].append(eval_result['preds'].detach().cpu().numpy())

            # Ensure gpu resources are released.
            del batch
            del eval_result
            
            if device == 'cuda' or device == 'gpu':
                torch.cuda.empty_cache()
            
            # Early stop for debugging
            # break
    
    return (
        np.concatenate(result['truth']),
        np.concatenate(result['score'])
    )


if __name__ == '__main__':
    model_path = 'lightning_checkpoints/LRT_v5_filter/version_3136300/checkpoints/last.ckpt'
    save = Path('0.3_900')
    save.mkdir(exist_ok=True, parents=True)

    model = load_model(
        model_path
    )

    truth, score = test_model(
        model
    )

    plt.rcParams.update({'font.size': 16})
    
    # Score dist
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_xlabel('Score')
    ax.set_ylabel('Arbitrary Scale')
    
    ax.hist(
        score[truth], log=True, histtype='step', bins=20, label="True Edges"
    )
    ax.hist(score[~truth], log=True, histtype='step', bins=20, label="False Edges")

    ax.legend()

    fig.savefig(save/'score_dist.pdf')

    # AUC
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    fpr, tpr, _ = roc_curve(truth, score)
    auc_score = auc(fpr, tpr)
    
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, label=f'AUC={auc_score:.4f}')
    ax.legend()

    fig.savefig(save/'auc.pdf')

