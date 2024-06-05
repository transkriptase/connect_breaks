import h5py
import numpy as np
from fillmissing import fill_missing
from cleaning import clean_and_validate_data
from loadh5 import load_h5_data
from loadh5 import print_attributes

filename = "C+1_1_0.h5"

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

def generate_connected_track_name(track_chain):
    return "_".join(track_chain)

def connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names, frame_count):
    connections = []
    remaining_broken_tracks = broken_tracks.copy()
    connected_tracks = set()
    completed_tracks = []
    track_chains = {track: [track] for track in broken_tracks}

    while remaining_broken_tracks:
        new_connections = []
        for track1_name, (start1, end1) in list(remaining_broken_tracks.items()):
            if track1_name in connected_tracks:
                continue
            best_candidate = None
            best_suitability_score = float('inf')
            for track2_name, (start2, end2) in not_real_tracks.items():
                for frame_offset in range(-10, frame_threshold + 1):  # Allowing overlap with negative frame differences
                    frame_diff = (start2 - end1) + frame_offset
                    if -(frame_threshold / 2) < frame_diff <= frame_threshold:
                        if track1_name in track_names:
                            track1_idx = track_names.index(track1_name)
                        else:
                            continue
                        track2_idx = track_names.index(track2_name)
                        track1_locations = filled_locations[:, :, :, track1_idx]
                        track2_locations = filled_locations[:, :, :, track2_idx]
                        last_point_track1 = track1_locations[end1, 0]
                        first_point_track2 = track2_locations[start2, 0]
                        spatial_distance = calculate_distance(last_point_track1, first_point_track2)
                        if spatial_distance <= distance_threshold:
                            points_in_radius = [track2_locations[start2 + i, 0] for i in range(-radius, radius + 1) if 0 <= start2 + i < filled_locations.shape[0]]
                            if circle_check(last_point_track1, points_in_radius, radius):
                                suitability_score = frame_diff + spatial_distance
                                if suitability_score < best_suitability_score:
                                    best_candidate = (track2_name, start2, end2)
                                    best_suitability_score = suitability_score
            if best_candidate:
                new_connections.append((track1_name, start1, end1, best_candidate[0], best_candidate[1], best_candidate[2]))
                not_real_tracks.pop(best_candidate[0])  # Remove connected track from not_real_tracks
                connected_tracks.add(track1_name)
                track_chains[track1_name].append(best_candidate[0])
                if best_candidate[2] == frame_count - 1:  # If the connected track ends at end_frame
                    completed_tracks.append(generate_connected_track_name(track_chains[track1_name]))

        if not new_connections:
            break
        for conn in new_connections:
            connections.append(conn)
            track1_name, start1, end1, track2_name, start2, end2 = conn
            # Update broken_tracks with new end frames
            if end2 < frame_count - 1:
                remaining_broken_tracks[track2_name] = (start2, end2)
            broken_tracks[track1_name] = (start1, end2)
        remaining_broken_tracks = {k: v for k, v in broken_tracks.items() if k in [conn[0] for conn in new_connections]}

    return connections, completed_tracks, track_chains

def complete_new_tracks(new_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names, frame_count, completed_tracks):
    while new_tracks:
        connections, additional_completed_tracks, track_chains = connect_broken_tracks(new_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names, frame_count)
        completed_tracks.extend(additional_completed_tracks)
        new_tracks = {k: v for k, v in create_new_tracks(connections, track_chains).items() if v[1] != frame_count - 1}
        track_names.extend(new_tracks.keys())

    return completed_tracks

def create_new_tracks(connections, track_chains):
    new_tracks = {}
    for track1_name, start1, end1, track2_name, start2, end2 in connections:
        new_track_name = generate_connected_track_name(track_chains[track1_name])
        new_tracks[new_track_name] = (start1, end2)
    return new_tracks

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
    print(len(broken_tracks))
    print("NOT REAL TRACKS", not_real_tracks)
    print(len(not_real_tracks))
    print("NOT BROKEN TRACKS", not_broken_tracks)
    print(len(not_broken_tracks))
    
    connected_tracks, completed_tracks, track_chains = connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names, frame_count)
    print("CONNECTED TRACKS")
    print(len(connected_tracks))
    for track1_name, start1, end1, track2_name, start2, end2 in connected_tracks:
        print(f"Connected: {track1_name} (end frame {end1}) --> {track2_name} (start frame {start2}, end frame {end2})")
    
    new_tracks = create_new_tracks(connected_tracks, track_chains)
    print("NEW TRACKS")
    print(len(new_tracks))
    for new_track_name, (start_frame, end_frame) in new_tracks.items():
        print(f"{new_track_name}: start frame {start_frame}, end frame {end_frame}")
    
    completed_tracks = complete_new_tracks(new_tracks, not_real_tracks, frame_threshold, distance_threshold, radius, filled_locations, track_names, frame_count, completed_tracks)

    print(" ")
    print("COMPLETED TRACKS")
    for completed_track in completed_tracks:
        print(completed_track)
    print(len(completed_tracks))
    print("NEW NOT REAL TRACKS", not_real_tracks)
    print(len(not_real_tracks))
