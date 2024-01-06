import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob, os
import imageio
import csv
from .config import Config

def load_body_region_points(config):
    """ Loads boady region points from CSV file.
    
    This function loads the CSV path stored in the cow_centroid_model_data attribute 
    of the Config object and returns a numpy array of the body region points.
    The training labels for the CNNs have two spine points so those are averaged.
    Missing values are relaced with -1.

    Args:
        config: Config object

    Returns:
        ndarray: 2D numpy array of body region points in order of config.body_region_points
    """
    # Read the CSV file, skipping the first line but keeping the second for headers
    with open(config.cow_centroid_model_data, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the scorer line
        headers = next(reader)  # Second line for column headers
        next(reader)  # Skip the coords line

        headers = [header.lower() for header in headers]
        point_indices = {}

        for point in config.body_region_points:
            if point == 'spine': # Handle two Spine points 
                point_indices[point] = [
                    headers.index('spinefront'), headers.index('spineback'),
                    headers.index('spinefront') + 1, headers.index('spineback') + 1
                ]
            else:
                point_indices[point] = [
                    headers.index(point), headers.index(point) + 1 
                ]
        # Process each row in the CSV file
        rows = []
        for row in reader:
            values = []
            for point, indices in point_indices.items():
                if point == 'spine':
                    # Combine two spine points into a single point computed as the average of the two 
                    x = (float(row[indices[0]]) + float(row[indices[1]])) / 2 if row[indices[0]] and row[indices[1]] else -1
                    y = (float(row[indices[2]]) + float(row[indices[3]])) / 2 if row[indices[2]] and row[indices[3]] else -1
                    x = -1 if x < 0 else x
                    y = -1 if y < 0 else y
                    values.extend([x, y])
                else:
                    x = float(row[indices[0]]) if row[indices[0]] else -1
                    y = float(row[indices[1]]) if row[indices[1]] else -1
                    values.extend([x, y])
            rows.append(values)

    return np.array(rows)

def generate_cow_centroid_model(config):
    """  Generates a model for estimating cow centroid location from body region points.

    Args:
        config: Config object

    Returns:
        part_gaussian (Dict) : Dictionary with body region point partnames as keys
                                 and a list with 2 nested lists containing the mean and standard
                                 deviation of the differences between the centroid and
                                 each point as values. [[x_mean, y_mean], [x_std, y_std]] 
    """
    all_pts = load_body_region_points(config)
	# Filters out any images that does not have all parts labeled.
	# All labels are needed to get an accurate centroid.
    pts = all_pts[np.min(all_pts, axis = 1) > 0] 
    # Extract x values by grabbing every other column starting from 0
    pts_x = pts[:,::2]
    # Extract y values by grabbing every other column starting from 1
    pts_y = pts[:,1:][:,::2]
    mid_values = []
    # Calculate cow centroid for each frame
    for i in range(len(pts_x)):
        cur_pts_x = pts_x[i,:]
        cur_pts_y = pts_y[i,:]
        x_mid, y_mid = np.mean(cur_pts_x), np.mean(cur_pts_y)
        mid_values.append([x_mid, y_mid])	
    mid_values = np.array(mid_values)
    # Empty dictionary to store values for each body part
    part_gaussian = {}
    # For each body region point 
    for i, part in enumerate(config.body_region_points):
        # Calculate the difference between the centroid and each point
        dif_x = pts_x[:,i] - mid_values[:,0]
        dif_y = pts_y[:,i] - mid_values[:,1]
        # Calculate the mean and standard deviation of the differences
        mu = [np.mean(dif_x), np.mean(dif_y)]
        std = [np.std(dif_x), np.std(dif_y)]
        # Store the mean and standard deviation in a dictionary with the bodypart's name as the key
        part_gaussian[part] = [mu, std]
    return part_gaussian
