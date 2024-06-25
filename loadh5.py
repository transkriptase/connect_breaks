import h5py
import numpy as np
from fillmissing import fill_missing
from cleaning import clean_and_validate_data

#filename = "C+1_1_0.h5"
#filename = "C-6_2.h5"

def load_h5_data(filename):
    with h5py.File(filename, "r") as f:
        track_names = [n.decode() for n in f["track_names"][:]]
        locations = f["tracks"][:].T
        frame_count, node_count, _, instance_count = locations.shape
        node_names = [n.decode() for n in f["node_names"][:]]
        
        print("Dataset names:", list(f.keys()))
        print("\n===== TRACK NAMES =====")
        print(track_names)
        f.visititems(print_attributes)
    
    print(frame_count)
    print(instance_count)
    return frame_count, node_count, instance_count, locations, track_names, node_names

def print_attributes(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("  {}: {}".format(key, val))
    print("  Type: {}".format(type(obj)))
    if isinstance(obj, h5py.Dataset):
        print("  Shape: {}".format(obj.shape))
        print("  Data Type (dtype): {}".format(obj.dtype))
    print("==================================")
