import os
import pathlib

import numba
import numpy as np
import torch
from torch.utils.data import Dataset

from pdp.dataset.replay_buffer import ReplayBuffer
from pdp.utils.data import dict_apply
from pdp.utils.normalizer import LinearNormalizer
import joblib
import h5py

# Get the top-level directory of the project
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]


@numba.jit(nopython=True)
def create_indices(motion_starts, motion_lengths, exclude_ids, sequence_length, pad_before=0, pad_after=0):
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)
    indices = list()
    for i in range(len(motion_starts)):
        if i in exclude_ids:
            continue

        start_idx = motion_starts[i]
        episode_length = motion_lengths[i]
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx
            ])

    indices = np.array(indices)
    return indices


class DiffusionPolicyDatasetPHC(Dataset):
    def __init__(self, 
            data_path, horizon=1, pad_before=0, pad_after=0, cache_data=False):
        self.keys = ['obs', 'action']
        self.meta_keys = ['motion_fname']
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.cache_data = cache_data



        # zarr_path = os.path.join(PROJECT_DIR, zarr_path)
        meta_data = joblib.load(f'{data_path}/phc_act_amass_train_upright_metadata.pkl')
        self.normalizer_path = f'{data_path}/normalizer_params.pt'
        self.h5path =  f'{data_path}/phc_act_amass_train_upright.h5'
        self.h5file = None

        self.cache = {}

        if self.cache_data:
            print('Loading data into cache ========================')
            with h5py.File(self.h5path, 'r') as h5f:
                for key in ['clean_action', 'pdp_obs']:
                    self.cache[key] = h5f[key][()]
            print('Loading data complete ========================')

        motion_lengths = np.concatenate( [ml for ml in meta_data['motion_lengths']])
        self.num_motions = len(motion_lengths) - len(meta_data['exclude_ids'])
        motion_starts = np.cumsum(motion_lengths)
        motion_starts = np.insert(motion_starts, 0, 0)
        motion_starts = motion_starts[:-1]

        self.load_normalizer()
 
        self.indices = create_indices(
            motion_starts,
            motion_lengths,
            meta_data['exclude_ids'],
            sequence_length=self.horizon, 
            pad_before=pad_before, 
            pad_after=pad_after
        )
    
    def load_normalizer(self):
        saved_state = torch.load(self.normalizer_path)
        self.normalizer = LinearNormalizer()
        self.normalizer._manual_load_dict(saved_state)
        # self.normalizer.load_state_dict(saved_state)
        return 

    def get_normalizer(self):
        return self.normalizer

    @property
    def num_episodes(self):
        return self.num_motions

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.sample_sequence(idx)
        data = {    
            'obs': sample['pdp_obs'],           # T, D_o
            'action': sample['clean_action'],     # T, D_a
        }
        data = dict_apply(data, torch.from_numpy)
        data = dict_apply(data, lambda x: x.to(torch.float32))
        return data

    def get_validation_dataset(self):
        raise NotImplementedError

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx  = self.indices[idx]
        result = dict()

        if self.cache_data:
            data_source = self.cache

        else:
            if self.h5file is None:
                # This is important for multiprocessing with DataLoader
                self.h5file = h5py.File(self.h5path, 'r')
            data_source=self.h5file    

        for key in ['clean_action', 'pdp_obs']:
            sample = data_source[key][buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
                data = np.zeros((self.horizon, *sample.shape[1:]), dtype=sample.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.horizon:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample

            result[key] = data

        return result

    def get_episode_iterator(self):
        print('get_episode_iterator::: Why are we here')
        raise NotImplementedError
        # for ep_idx in range(self.num_episodes):
        #     start = 0
        #     if ep_idx > 0: start = self.episode_ends[ep_idx-1]
        #     end = self.episode_ends[ep_idx]
        #     ep_data = dict()
        #     for key in self.keys:
        #         ep_data[key] = self.replay_buffer[key][start:end]

        #     for key in self.meta_keys:
        #         ep_data[key] = self.replay_buffer.meta[key][ep_idx]
            
        #     yield ep_data
