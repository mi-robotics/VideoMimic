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
    inidicies_to_motion = list()
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
            inidicies_to_motion.append(i)
    return labels, inidicies_to_motion

class DiffusionPolicyDatasetPHCVidmimic(DiffusionPolicyDatasetPHC):
    def __init__(self, 
            data_path, horizon=1, pad_before=0, pad_after=0, cache_data=False, context_length=0):
        super().__init__(data_path, horizon, pad_before, pad_after, cache_data, load_keys=['clean_action', 'pdp_obs'])

        self.image_mappings = joblib.load(f'{data_path}/vidmimic_image_mappings.pkl')
        if os.path.exists(f'{data_path}/vidmimic_image_emb_mappings.pkl'):
            self.image_emb_mappings = joblib.load(f'{data_path}/vidmimic_image_emb_mappings.pkl')
        else:
            self.image_emb_mappings = None
        motion_keys = np.concatenate( [kn for kn in self.meta_data['key_names']])
        self.label_map, self.inidicies_to_motion = create_idx_label_map(
            motion_keys,
            self.motion_starts,
            self.motion_lengths,
            self.exclude_ids,
            self.horizon,
            self.pad_before, self.pad_after)

        assert len(self.label_map) == len(self.indices)
        assert context_length > 0
        self.context_length = context_length

    def __getitem__(self, idx):
        sample = self.sample_sequence(idx)
        motion_key = self.label_map[idx]

        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx  = self.indices[idx]
        motion_start_idx = self.motion_starts[self.inidicies_to_motion[idx]]

        if self.image_emb_mappings is not None:
            images_emb = self.image_emb_mappings[motion_key]
            start_frame_index =( buffer_start_idx + self.context_length)-motion_start_idx
            end_frame_index = buffer_end_idx - motion_start_idx

            images_emb = images_emb[[start_frame_index, end_frame_index]].copy()
            images = None
        else:
            raise NotImplementedError("Image embeddings must be provided currently")
            images = None # captions[cap_id]
            images_emb = None

        data = {    
            'obs': sample['pdp_obs'],           # T, D_o
            'action': sample['clean_action'],     # T, D_a
            'image_emb': images_emb,
        }
        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))

        return data




