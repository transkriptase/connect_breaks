import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from fillmissing import fill_missing
from cleaning import clean_and_validate_data
from loadh5 import load_h5_data

directory = "h5"
output_directory = "output"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

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

def detect_flips(positions, threshold=20):
    flips = []
    for i in range(1, len(positions)):
        prev_frame, prev_pos = positions[i - 1]
        curr_frame, curr_pos = positions[i]
        if calculate_distance(curr_pos, prev_pos) > threshold:
            flips.append(curr_frame)
    return flips

def correct_flips(positions, flips):
    corrected_positions = positions.copy()
    for flip_frame in flips:
        for i, (frame_idx, pos) in enumerate(corrected_positions):
            if frame_idx == flip_frame:
                corrected_positions[i] = (frame_idx, corrected_positions[i - 1][1])
    return corrected_positions

def get_instance_positions(locations, track_idx):
    positions = []
    for frame_idx in range(locations.shape[0]):
        if not np.all(np.isnan(locations[frame_idx, :, :, track_idx])):
            positions.append((frame_idx, locations[frame_idx, :, :, track_idx]))
    return positions

def update_locations_with_corrected_positions(locations, track_idx, corrected_positions):
    for frame_idx, corrected_pos in corrected_positions:
        locations[frame_idx, :, :, track_idx] = corrected_pos
    return locations

def circle_check(last_point, points, radius):
    for point in points:
        if calculate_distance(last_point, point) < radius:
            return True
    return False

def calculate_total_distance(locations):
    total_distances = {}
    frame_count, _, _, track_count = locations.shape
    
    for track_idx in range(track_count):
        x_data = locations[:, 0, 0, track_idx]
        y_data = locations[:, 0, 1, track_idx]
        
        # Calculate total distance traveled
        distances = np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2)
        total_distance = np.nansum(distances)
        
        # Store total distance for each track
        total_distances[track_idx] = total_distance
    
    return total_distances
    
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

    # Remove completed tracks from broken_tracks and not_real_tracks
    for completed_track in completed_tracks:
        for track in completed_track.split('_'):
            if track in broken_tracks:
                del broken_tracks[track]
            if track in not_real_tracks:
                del not_real_tracks[track]

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

def calculate_metrics(locations, track_names):
    metrics = {}
    for track_idx, track_name in enumerate(track_names):
        track_data = locations[:, :, :, track_idx]
        x_data = track_data[:, 0, 0]
        y_data = track_data[:, 0, 1]
        
        # Calculate total distance traveled
        total_distance = np.nansum(np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2))
        
        # Calculate average speed
        average_speed = total_distance / (np.count_nonzero(~np.isnan(x_data)) - 1)
        
        metrics[track_name] = {
            "total_distance": total_distance,
            "average_speed": average_speed
        }
    return metrics

def calculate_metrics_for_connected_tracks(locations, track_chains):
    metrics = {}
    for new_track_name, track_chain in track_chains.items():
        x_data = np.concatenate([locations[:, 0, 0, track_names.index(track_name)] for track_name in track_chain])
        y_data = np.concatenate([locations[:, 0, 1, track_names.index(track_name)] for track_name in track_chain])
        
        # Calculate total distance traveled
        total_distance = np.nansum(np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2))
        
        # Calculate average speed
        average_speed = total_distance / (np.count_nonzero(~np.isnan(x_data)) - 1)
        
        metrics[new_track_name] = {
            "total_distance": total_distance,
            "average_speed": average_speed
        }
    return metrics

def plot_tracks(locations, track_names, filename):
    plt.figure(figsize=(10, 8))
    for track_idx, track_name in enumerate(track_names):
        track_data = locations[:, :, :, track_idx]
        x_data = track_data[:, 0, 0]
        y_data = track_data[:, 0, 1]
        plt.plot(x_data, y_data)
    plt.title(f"Tracks for {filename}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_results.png")
    plt.savefig(output_path)
    plt.close()

def plot_average_speeds(all_metrics):
    avg_speeds = {}
    for filename, metrics in all_metrics.items():
        total_avg_speed = np.mean([data['average_speed'] for data in metrics.values()])
        avg_speeds[filename] = total_avg_speed
    
    plt.figure(figsize=(12, 6))
    plt.bar(avg_speeds.keys(), avg_speeds.values())
    plt.xlabel("Plate")
    plt.ylabel("Average Speed (units)")
    plt.title("Average Speed of Each Plate")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "average_speeds.png"))
    plt.close()


if __name__ == "__main__":
    all_metrics = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            filepath = os.path.join(directory, filename)
            try:
                with h5py.File(filepath, "r") as f:
                    track_names = [n.decode() for n in f["track_names"][:]]
                    locations = f["tracks"][:].T
                    frame_count, node_count, _, track_count = locations.shape
            except FileNotFoundError:
                print(f"File '{filepath}' not found.")
                continue

            frame_count, node_count, instance_count, locations, track_names, node_names = load_h5_data(filepath)
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
            
            not_real_tracks = {k: v for k, v in track_start_end_frames.items() if k not in tracks_starting_at_zero and (v[1] - v[0] >= 10)}
            not_broken_tracks = {k: v for k, v in tracks_starting_at_zero.items() if v[1] == frame_count - 1}
            broken_tracks = {k: v for k, v in tracks_starting_at_zero.items() if v[1] != frame_count - 1}

            print(f"Processing file: {filename}")
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
            print("COMPLETED TRACKS")
            print(len(completed_tracks))
            for completed_track in completed_tracks:
                print(completed_track)

            print(" ")
            print("REMAINING NEW TRACKS")
            for new_track_name, (start_frame, end_frame) in new_tracks.items():
                if new_track_name not in completed_tracks:
                    print(f"{new_track_name}: start frame {start_frame}, end frame {end_frame}")
            print(" ")

            # Calculate and store metrics for both individual and connected tracks
            individual_metrics = calculate_metrics(filled_locations, track_names)
            connected_metrics = calculate_metrics_for_connected_tracks(filled_locations, track_chains)
            
            # Combine both metrics
            combined_metrics = {**individual_metrics, **connected_metrics}
            all_metrics[filename] = combined_metrics

            # Plot and save the results
            plot_tracks(filled_locations, track_names, filename)






def plot_tracks(locations, track_names, filename):
    plt.figure(figsize=(10, 8))
    for track_idx, track_name in enumerate(track_names):
        track_data = locations[:, :, :, track_idx]
        x_data = track_data[:, 0, 0]
        y_data = track_data[:, 0, 1]
        plt.plot(x_data, y_data)
    plt.title(f"Tracks for {filename}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_results.png")
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    all_metrics = {}
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            filepath = os.path.join(directory, filename)
            try:
                with h5py.File(filepath, "r") as f:
                    track_names = [n.decode() for n in f["track_names"][:]]
                    locations = f["tracks"][:].T
                    frame_count, node_count, _, track_count = locations.shape
            except FileNotFoundError:
                print(f"File '{filepath}' not found.")
                continue

            frame_count, node_count, instance_count, locations, track_names, node_names = load_h5_data(filepath)
            filled_locations = fill_missing(locations)
            cleaned_dataset = clean_and_validate_data(filled_locations)

            # Calculate total distances traveled by each track
            total_distances = calculate_total_distance(filled_locations)

            # Print or analyze total distances for each termite (track)
            for track_idx, distance in total_distances.items():
                print(f"Track {track_idx + 1}: Total Distance Traveled = {distance}")

            # Continue with your existing analysis, plotting, and metrics calculations...

            # Store and plot the results
            plot_tracks(filled_locations, track_names, filename)
