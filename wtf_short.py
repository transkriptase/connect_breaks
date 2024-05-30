import h5py
import numpy as np

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2)

def find_start_end_frames(track_idx, locations):
    """Find start and end frames for a track."""
    start_frame, end_frame = None, None
    frame_count = locations.shape[0]
    for frame_idx in range(frame_count):
        if not np.all(np.isnan(locations[frame_idx, :, :, track_idx])):
            end_frame = frame_idx
            if start_frame is None:
                start_frame = frame_idx
    return start_frame, end_frame

def connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, filled_locations, track_names):
    connections = []

    for track1_name, (start1, end1) in broken_tracks.items():
        best_candidate = None
        best_suitability_score = float('inf')

        for track2_name, (start2, end2) in not_real_tracks.items():
            frame_diff = start2 - end1

            # Check if the frames are within the threshold
            if 0 < frame_diff <= frame_threshold:
                track1_idx = track_names.index(track1_name)
                track2_idx = track_names.index(track2_name)

                # Extract track locations
                track1_locations = filled_locations[:, :, :, track1_idx]
                track2_locations = filled_locations[:, :, :, track2_idx]

                # Calculate distance between the last point of track1 and the first point of track2
                last_point_track1 = track1_locations[end1]
                first_point_track2 = track2_locations[start2]
                spatial_distance = calculate_distance(last_point_track1, first_point_track2)

                # Check if the spatial distance is within the threshold
                if spatial_distance <= distance_threshold:
                    suitability_score = frame_diff + spatial_distance  # Consider both frame and spatial distance

                    if suitability_score < best_suitability_score:
                        best_candidate = (track2_name, start2, end2)
                        best_suitability_score = suitability_score

        if best_candidate:
            connections.append((track1_name, start1, end1, best_candidate[0], best_candidate[1], best_candidate[2]))

    return connections

# Load the data from the H5 file
filename = "C+1_1_0.h5"

with h5py.File(filename, "r") as f:
    track_names = [n.decode() for n in f["track_names"][:]]
    locations = f["tracks"][:].T
    frame_count, node_count, _, track_count = locations.shape

filled_locations = np.nan_to_num(locations)

# Define thresholds
frame_threshold = 50  # Adjust frame threshold as needed
distance_threshold = 20  # Adjust distance threshold as needed

# Find start and end frames for all tracks
track_start_end_frames = {}
for track_idx in range(track_count):
    start_frame, end_frame = find_start_end_frames(track_idx, locations)
    if start_frame is not None and end_frame is not None:
        track_start_end_frames[track_names[track_idx]] = (start_frame, end_frame)

# Separate tracks into categories
broken_tracks = {k: v for k, v in track_start_end_frames.items() if v[1] != frame_count - 1}
not_real_tracks = {k: v for k, v in track_start_end_frames.items() if k not in broken_tracks}

# Connect broken tracks
connected_tracks = connect_broken_tracks(broken_tracks, not_real_tracks, frame_threshold, distance_threshold, filled_locations, track_names)

# Print connected tracks
print("CONNECTED TRACKS")
for track1_name, start1, end1, track2_name, start2, end2 in connected_tracks:
    print(f"Connected: {track1_name} (start frame {start1}, end frame {end1}) --> {track2_name} (start frame {start2}, end frame {end2})")
print(f"Total connected tracks: {len(connected_tracks)}")
