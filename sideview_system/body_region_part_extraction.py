from .utils import compute_distance
import numpy as np
from .config import Config
from skimage.measure import regionprops, label

class bodypart():
	"""
    Attributes:
    - name (str): The name of the body part.
    - pt (ndarray): Point location stored in [row, col]. Upsamples to original image size.
    - score (float): Confidence score of detection
    - orig_pts (list): When bodyparts are merged to form a single bodypart, this list stores the original locations of the merged parts.
    - orig_scores (list): When bodyparts are merged to form a single bodypart, this list stores the original confidence scores of the merged parts.
    - pt_center (ndarray): Cow centroid location estimated from body part location
    - pt_center_std (float): Standard deviation of the cow centroid location estimate
    """
	def __init__(self, 
				 name, 
				 prop = -1,
				 pt = None, 
				 score = None,
				 orig_pts = [],
				 orig_scores = [],
				 pt_center = [-1,-1],
				 pt_center_std = -1):
		self.name = name
		if prop != -1:
			self.init_from_prop(prop)
		else:
			self.pt = pt
			self.score = score
			self.orig_pts = orig_pts
			self.orig_scores = orig_scores
		self.pt_center = np.array(pt_center)
		self.pt_center_std = pt_center_std

	def __repr__(self):
		""" Returns a string representation of the object in the format partname_x_y_confidence_score"""
		return self.name + str('_') + str(self.pt[0])[:5] + ' ' + str(self.pt[1])[:5] + str('_') + str(self.score)

	def init_from_prop(self, prop):
		""" Initializes the body part from a skimage.measure.regionprops object. """
		# Multiply by 8 to upscale from scoremap space to original image space
    	# Add 4 to shift from the top-left corner of the corresponding 8x8 patch to center 
		# 8x8 patch because connectivity = 2 was passed to skimage.label which defines 8 neighbors
		# skimage uses the convention (row, col) for coordinates!
		self.pt = np.array(prop.centroid).astype(int)*8+4
		self.score = prop.mean_intensity
		self.orig_pts = prop.coords*8+4
		self.orig_scores = prop.intensity_image

def merge_bodyparts(parts, spine = False):
	""" Merges multiple bodypart objects into a single bodypart object.

    Parameters:
    - parts (list of bodypart): The list of bodypart objects detected for the same body part.
    - spine (bool): Indicates whether the parts being merged are spine parts. Defaults to False.

    Returns:
    - bodypart: A single, consolidated bodypart object representing the merged detections.
    """
	# If only one part, and spine flag is false, return the part
	if len(parts) == 1 and not spine: 
		return parts[0]

	# If spine flag is true, create a new spine bodypart object 
	# Spine flag is only true when this function is called from merge_detected_spine_points(..) 
	if spine:
		new_part = bodypart(name = 'spine')
	# Otherwise, create a new part and carry over the name from the first part. 
	else:
		assert len(np.unique([x.name for x in parts])) == 1, 'two parts should be the same type'
		new_part = bodypart(name = parts[0].name)

	# Initialize the new part's attributes to be the mean of the parts being merged
	new_part.pt = np.mean(np.array([x.pt for x in parts]), axis = 0) # estimated location 
	new_part.score = np.mean([x.score for x in parts]) # confidence score
	new_part.pt_center = np.mean(np.array([x.pt_center for x in parts]), axis = 0) # estimated center point
	new_part.pt_center_std = np.mean([x.pt_center_std for x in parts]) # estimated center point standard deviation

	# Create empty lists to store the original locations and confidence scores of all the parts being merged
	temp_orig_pts, temp_orig_scores = [], []

	# For each part being merged, populate the lists with the original locations and confidence score
	for x in parts:
		temp_orig_pts.append(x.orig_pts)
		temp_orig_scores.append(x.score)

	# Sets the new part's original locations and confidence scores to the lists we just populated
	new_part.orig_pts = temp_orig_pts
	new_part.orig_scores = temp_orig_scores

	return new_part

def group_by_distance(parts, th=5):
    """
    Merges keypoints based on spatial proximity within a specified threshold.

    This function takes a list of 'bodypart' objects and groups them based on the 
    Euclidean distance between their coordinates. If the distance between two keypoints 
    is less than the specified threshold, they are considered to be part of the same group. 
    This is useful for consolidating keypoints that represent the same physical point but 
    have been detected multiple times.

    Args:
        parts (list of bodypart): A list of 'bodypart' objects, each with a 'pt' attribute representing coordinates.
        th (int, optional): The threshold distance for merging keypoints. Defaults to 5.

    Returns:
        List[List[int]]: A list of lists, where each sublist contains indices of parts that are 
                         close enough to be considered the same keypoint.
    """
    # Extract the point coordinates from each 'bodypart' object
    pts = [x.pt for x in parts]
    # Create a list of indices representing each keypoint
    idx_list = list(range(len(pts)))
    # Initialize a list to hold groups of merged keypoints
    pt_candidate_list = []

    # Iterate through all keypoints, grouping close ones
    while idx_list:
        # Start a new group with the first ungrouped keypoint
        idx = idx_list.pop(0)
        pt_candidate_list.append([idx])
        # Check remaining keypoints for proximity
        for i in idx_list:
            if compute_distance(pts[idx], pts[i]) < th:
                pt_candidate_list[-1].append(i)
		# Remove grouped keypoints from future consideration
        for k in pt_candidate_list[-1]:
            if k in idx_list:
	            idx_list.remove(k)
		
    return pt_candidate_list


def get_candidate_parts_from_cmap_channel(single_channel_cmap, partname, th = 0.6):
	label_map = label(single_channel_cmap > th, background = 0, connectivity = 2)
	# Get skimage properties for each labeled region
	# Stores bodypart centroid (estimated location of key point) and mean intensity of region (confidence score)
	props = regionprops(label_map, intensity_image = single_channel_cmap)
	init_parts, final_candidate_parts = [], []

	# Use each labeled region to create a bodypart object
	for prop in props:
		init_parts.append(bodypart(partname, prop = prop))

	grouped_part_candidates = group_by_distance(init_parts)

	for idxes in grouped_part_candidates:
		final_candidate_parts.append(merge_bodyparts([init_parts[x] for x in idxes]))

	return final_candidate_parts

def extract_body_region_parts(multi_channel_cmap, config, dlc_cfg, th = 0.6, merge_th = 5):
	body_region_parts = []
	all_point_names = [x.lower() for x in dlc_cfg["all_joints_names"]]

	for i, partname in enumerate(all_point_names):
		if partname in config.body_region_points + ['spinefront', 'spineback']:
			single_channel_cmap = multi_channel_cmap[:,:,i]
			parts = get_candidate_parts_from_cmap_channel(single_channel_cmap, partname, th)
			body_region_parts.extend(parts)

	other_parts = [x for x in body_region_parts if x.name not in ['spineback', 'spinefront']]
	spine_parts = [x for x in body_region_parts if x.name in ['spinefront', 'spineback']]

	if len(spine_parts) == 0: return other_parts

	grouped_spine_parts = group_by_distance(spine_parts, th = 300)
	for idxes in grouped_spine_parts:
		other_parts.append(merge_bodyparts([spine_parts[x] for x in idxes], spine = True))

	return other_parts