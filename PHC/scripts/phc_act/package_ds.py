import glob
import os
import sys
import pdb
import os.path as osp


import joblib
import numpy as np
import h5py
from tqdm import tqdm
from collections import defaultdict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/amass/pkls/amass_isaac_run_upright_slim.pkl")
    parser.add_argument("--exp_name", type=str, default="phc_kp_mcp_iccv")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--action_noise_std", type=float, default=0.05)
    args = parser.parse_args()



    add_action_noise = True
    action_noise_std = 0.05
    dataset_path = args.dataset_path
    motion_file_name = dataset_path.split("/")[-1].split(".")[0]
    exp_name = args.exp_name
   
    num_runs = args.num_runs

    assert exp_name == 'phc_kp_mcp_iccv'

    import h5py
    import glob
    import joblib
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict

    print("Done")
    # Aggregating the dataset into one file
    pkl_files = glob.glob(f"output/HumanoidIm/{exp_name}/phc_act/{motion_file_name}/*.pkl")

    # Use a defaultdict to store the metadata separately
    metadata_dump = defaultdict(list)

    # Open the HDF5 file once and keep it open for appending
    with h5py.File(f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}.h5', 'w') as hdf5_file:
        
        # Initialize HDF5 datasets with dummy data and maxshape
        # This assumes the first file has a representative structure and shape
        first_file_data = joblib.load(pkl_files[0])
        
        for key, value in first_file_data.items():
            if key in ['pdp_obs', "clean_action", "reset"]:
                if isinstance(value, list) and value:
                    first_chunk = np.concatenate(value)
                    # Create resizable datasets
                    hdf5_file.create_dataset(key, data=first_chunk, maxshape=(None,) + first_chunk.shape[1:], compression="gzip", compression_opts=9)
                else:
                    # Handle cases where value is already an array
                    hdf5_file.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:], compression="gzip", compression_opts=9)
            else:
                # Store metadata for later dumping
                metadata_dump[key] = value
                
        # Process the rest of the files
        for file in tqdm(pkl_files[1:]):
            file_data = joblib.load(file)
            
            for k, v in file_data.items():
                if k in ['pdp_obs', "clean_action", "reset"]:
                    if isinstance(v, list) and v:
                        chunk = np.concatenate(v)
                        dataset = hdf5_file[k]
                        # Resize and append the new chunk
                        old_size = dataset.shape[0]
                        new_size = old_size + chunk.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[old_size:new_size] = chunk
                    else:
                        dataset = hdf5_file[k]
                        old_size = dataset.shape[0]
                        new_size = old_size + v.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[old_size:new_size] = v
                else:
                    # Append metadata to a list to handle potential variations
                    metadata_dump[k].append(v)

    # Dump the metadata outside the HDF5 file
    # Note: You'll need to decide how to handle the aggregation of metadata if it varies
    joblib.dump(metadata_dump, f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}_metadata.pkl', compress=True)

    print("Dataset successfully written to HDF5 file, no longer dumping to a single .pkl")

