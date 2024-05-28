# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:20:20 2024

@author: okilic
"""

import sleap
import numpy as np

# Load the predictions file using raw string notation
labels = sleap.load_file(r"C:\Users\okilic\Desktop\post_sleap\C+1_1_0.slp")

# Parameters for connecting tracks
max_frame_gap = 10  # Maximum gap in frames to connect
max_distance = 50  # Maximum distance to connect instances

# Function to calculate distance between instances
def calculate_distance(inst1, inst2):
    if len(inst1.points) != len(inst2.points):
        return float('inf')  # Return a large distance if point counts do not match
    points1 = np.array([[pt.x, pt.y] for pt in inst1.points])
    points2 = np.array([[pt.x, pt.y] for pt in inst2.points])
    return np.linalg.norm(points1 - points2)

# Iterate through frames and connect broken tracks
for i in range(len(labels.labeled_frames)):
    frame = labels.labeled_frames[i]
    for inst1 in frame.instances:
        for j in range(i + 1, min(i + max_frame_gap, len(labels.labeled_frames))):
            next_frame = labels.labeled_frames[j]
            for inst2 in next_frame.instances:
                distance = calculate_distance(inst1, inst2)
                if distance < max_distance:
                    inst2.track = inst1.track

# Save the updated labels using raw string notation
labels.save(r"C:\Users\okilic\Desktop\post_sleap\C+1_1_0_connect.slp")
