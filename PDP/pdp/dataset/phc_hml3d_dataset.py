import os
import pathlib

import numba
import numpy as np
import torch
from torch.utils.data import Dataset

from pdp.dataset.replay_buffer import ReplayBuffer
from pdp.dataset.phc_dataset import DiffusionPolicyDatasetPHC
from pdp.utils.data import dict_apply
from pdp.utils.normalizer import LinearNormalizer
import joblib
import h5py

# Get the top-level directory of the project
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]



def create_idx_label_map(
        motion_labels,
        motion_starts, 
        motion_lengths, 
        exclude_ids, 
        sequence_length, 
        pad_before=0, 
        pad_after=0):

    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)
    labels = list()
    for i in range(len(motion_starts)):
        if i in exclude_ids:
            continue

        start_idx = motion_starts[i]
        episode_length = motion_lengths[i]
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            labels.append(motion_labels[i])

    return labels

class DiffusionPolicyDatasetPHCHml3d(DiffusionPolicyDatasetPHC):
    def __init__(self, 
            data_path, horizon=1, pad_before=0, pad_after=0, cache_data=False):
        super().__init__(data_path, horizon, pad_before, pad_after, cache_data)

        self.motion_labels = joblib.load(f'{data_path}/hml3d_labels.pkl')
        if os.path.exists(f'{data_path}/hml3d_embs.pkl'):
            self.clip_embs = joblib.load(f'{data_path}/hml3d_embs.pkl')
        else:
            self.clip_embs = None
        motion_keys = np.concatenate( [kn for kn in self.meta_data['key_names']])
        self.label_map = create_idx_label_map(
            motion_keys,
            self.motion_starts,
            self.motion_lengths,
            self.exclude_ids,
            self.horizon,
            self.pad_before, self.pad_after)

        assert len(self.label_map) == len(self.indices)

    def __getitem__(self, idx):
        sample = self.sample_sequence(idx)
        motion_key = self.label_map[idx]
        captions = self.motion_labels[motion_key]
        cap_id = np.random.randint(0, len(captions))
        
        caption = captions[cap_id]
        if self.clip_embs is not None:
            caption_emb = self.clip_embs[motion_key]
            caption_emb = caption_emb[cap_id]
        else:
            caption_emb = None

        data = {    
            'obs': sample['pdp_obs'],           # T, D_o
            'action': sample['clean_action'],     # T, D_a
            'caption': caption,
            'caption_emb': caption_emb,
        }
        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))

        return data




