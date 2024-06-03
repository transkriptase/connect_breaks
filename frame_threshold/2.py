import h5py
import numpy as np
from fillmissing import fill_missing
from cleaning import clean_and_validate_data

filename = "C+1_1_0.h5"

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

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def find_start_end_frames(track_idx, locations):
    start_frame, end_frame = None, None
    frame_count = locations.shape[0]
    for frame_idx in range(frame_count):
        if not np.all(np.isnan(locations[frame_idx, :, :, track_idx])):
            end_frame = frame_idx
            if start_frame is None:
                start_frame = frame_idx
    return start_frame, end_frame

def circle_check(last_point, points, radius):
    for point in points:
        if calculate_distance(last_point, point) < radius:
            return True
    return False

def connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names):
    connections = []
    for track1_name, (start1, end1) in broken_tracks.items():
        best_candidate = None
        best_suitability_score = float('inf')
        for track2_name, (start2, end2) in not_real_tracks.items():
            frame_diff = start2 - end1
            if -(frame_threshold/2) < frame_diff <= frame_threshold:
                track1_idx = track_names.index(track1_name)
                track2_idx = track_names.index(track2_name)
                track1_locations = filled_locations[:, :, :, track1_idx]
                track2_locations = filled_locations[:, :, :, track2_idx]
                last_point_track1 = track1_locations[end1, 0]  
                first_point_track2 = track2_locations[start2, 0]  
                spatial_distance = calculate_distance(last_point_track1, first_point_track2)
                if spatial_distance <= distance_threshold:
                    points_in_radius = [track2_locations[start2 + i, 0] for i in range(-radius, radius+1) if 0 <= start2 + i < filled_locations.shape[0]]
                    if circle_check(last_point_track1, points_in_radius, radius):
                        suitability_score = frame_diff + spatial_distance
                        if suitability_score < best_suitability_score:
                            best_candidate = (track2_name, start2)
                            best_suitability_score = suitability_score
        if best_candidate:
            connections.append((track1_name, end1, best_candidate[0], best_candidate[1]))
    return connections

if __name__ == "__main__":
    try:
        with h5py.File(filename, "r") as f:
            track_names = [n.decode() for n in f["track_names"][:]]
            locations = f["tracks"][:].T
            frame_count, node_count, _, track_count = locations.shape
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        exit()
    
    frame_count, node_count, instance_count, locations, track_names, node_names = load_h5_data(filename)
    filled_locations = fill_missing(locations)
    cleaned_dataset = clean_and_validate_data(filled_locations)
    
    frame_threshold = 100  # Increased from 50
    distance_threshold = 2000  # Pixel size in micrometers
    radius = 90  # Adjusted for broader checking

    track_start_end_frames = {}
    tracks_starting_at_zero = {}
    for track_idx in range(track_count):
        start_frame, end_frame = find_start_end_frames(track_idx, locations)
        if start_frame is not None and end_frame is not None:
            track_start_end_frames[track_names[track_idx]] = (start_frame, end_frame)
            if start_frame == 0:
                tracks_starting_at_zero[track_names[track_idx]] = (start_frame, end_frame)
    
    not_real_tracks = {k: v for k, v in track_start_end_frames.items() if k not in tracks_starting_at_zero}
    not_broken_tracks = {k: v for k, v in tracks_starting_at_zero.items() if v[1] == frame_count - 1}
    broken_tracks = {k: v for k, v in tracks_starting_at_zero.items() if v[1] != frame_count - 1}
    
    print("BROKEN TRACKS", broken_tracks)
    print("NOT REAL TRACKS", not_real_tracks)
    print("NOT BROKEN TRACKS", not_broken_tracks)
    
    connected_tracks = connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names)
    print("CONNECTED TRACKS")
    print(len(connected_tracks))
    for track1_name, end1, track2_name, start2 in connected_tracks:
        print(f"Connected: {track1_name} (frame {end1}) --> {track2_name} (frame {start2})")
