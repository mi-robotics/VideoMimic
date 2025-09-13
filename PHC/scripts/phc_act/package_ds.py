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

    # Use a defaultdict(list) to ensure all values are lists
    metadata_dump = defaultdict(list)
    hdf5_datasets = {} # Dictionary to hold h5py dataset objects

    # Open the HDF5 file once and keep it open for appending
    with h5py.File(f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}.h5', 'w') as hdf5_file:
        
        # Process the files one by one
        for i, file in enumerate(tqdm(pkl_files)):
            file_data = joblib.load(file)
            
            for k, v in file_data.items():
                if k in ['pdp_obs', "clean_action", "reset", 'motion_lengths']:
                    if isinstance(v, list) and v:
                        chunk = np.concatenate(v)
                    else:
                        chunk = v
                    
                    if i == 0:
                        # Create resizable datasets for the first file
                        hdf5_datasets[k] = hdf5_file.create_dataset(
                            k, data=chunk, maxshape=(None,) + chunk.shape[1:], 
                            compression="gzip", compression_opts=9
                        )
                    else:
                        # Resize and append for subsequent files
                        dataset = hdf5_datasets[k]
                        old_size = dataset.shape[0]
                        new_size = old_size + chunk.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[old_size:new_size] = chunk
                else:
                    # Always append metadata to a list
                    metadata_dump[k].append(v)

    # Dump the aggregated metadata outside the HDF5 file
    # Note: You may need to handle the aggregation of metadata after the loop,
    # e.g., for 'running_mean', you might want the last value or the average.
    for key, value in metadata_dump.items():
        if key == "running_mean":
            metadata_dump[key] = value[-1] # or np.mean(value, axis=0)
        # Add other aggregation logic here for other metadata keys
        
    joblib.dump(metadata_dump, f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}_metadata.pkl', compress=True)

    print("Dataset successfully written to HDF5 file, no longer dumping to a single .pkl")

