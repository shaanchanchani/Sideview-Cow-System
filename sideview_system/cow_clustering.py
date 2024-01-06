
from .config import *
from .cow_centroid_model import generate_cow_centroid_model
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import MeanShift



# These functions are in utils.py as well
# Necessary to include them here to avoid circular import, utils.py uses the cow class in draw_post_processing(...)
def compute_distance(p0, p1):
	d = np.sqrt(np.sum((np.array(p0) - np.array(p1))**2))
	return d

def compute_angle(l1, l2):
	# if two points are very close to each other
	if compute_distance(l1, l2) < 1: return 0
	return np.arccos(np.dot(l1, l2)/norm(l1)/norm(l2))

class cow():
	"""    
    Attributes:
    partnames (List[str]): List of names of the body parts.
    part_dict (Dict[str, List[bodypart]]): Dictionary mapping part names to lists of bodypart objects.
    part_idxes (Dict[str, List[int]]): Dictionary mapping part names to lists of their indices.
    count (Dict[str, int]): Dictionary mapping part names to their counts.
    weighted_center (List[int]): (row,col) coordinates of the weighted center of the cow.
	
    """
	def __init__(self, config, parts=[], partnames=[]):
		self.parts = []
		self.partnames = config.body_region_points
		self.part_dict = {}
		self.part_idxes = {}
		self.count = {}
		self.weighted_center = [-1, -1]
		for partname in self.partnames:
			self.part_dict[partname] = []
			self.part_idxes[partname] = []
			self.count[partname] = 0
		self.add_parts(parts)

	def __getitem__(self, key):
		""" Allows cow objects to be indexed by part name """
		idx = self.all_names.index(key)
		return self.all_parts[key]

	def __repr__(self):
		""" Returns a string representation of the object in the format cow_row_col with coordinates of the weighted center """
		return 'cow_' + str(self.weighted_center[0])[:5] + ' ' + str(self.weighted_center[1])[:5]

	def __len__(self):
		""" Returns the number of unique part names in the cow object """
		return len(self.all_names)

	def add_parts(self, parts, compute_center = True):
		"""
		Used to initialize and update:
			- part_dict: A dictionary mapping part names to lists of bodypart objects.
			- part_idxes: A dictionary mapping part names to lists of their indices.
			- count: A dictionary mapping part names to their counts.
		Parameters:
		parts (List[bodypart]): A list of bodypart objects to be added to the cow.
		compute_center (bool): A flag indicating whether to recompute the cow's center after adding the parts. Defaults to True.
		"""
		for i, part in enumerate(parts):
			partname = part.name
			if partname not in self.part_dict:
				self.part_dict[partname] = [part]
				self.part_idxes[partname] = [i]
				self.count[partname] = len(self.part_dict[partname])
			else:
				self.part_dict[partname].append(part)
				self.part_idxes[partname].append(i)
				self.count[partname] = len(self.part_dict[partname])
		if compute_center:
			self.estimate_cow_center()
		self.ravel()

	def delete_parts(self, partname, idx):
		self.part_dict[partname].pop(idx)
		l = len(self.part_dict[partname])
		self.count[partname] = l
		self.part_idxes[partname] = set(range(l))

	def ravel(self):
		"""
		Sorts the body parts by confidence score and flattens the part_dict, part_idxes, and count attributes.

		Used to initialize and update:
			- all_pts: A list of all the points of the body parts.
			- all_names: A list of all the names of the body parts.
			- all_parts: A list of all the bodypart objects.
		"""
		self.cow_parts_sort()
		all_pts, all_names, all_parts = [], [], []
		for partlist in self.part_dict.values():
			for part in partlist[:1]:
				all_pts.append(part.pt)
				all_names.append(part.name)
				all_parts.append(part)
		self.all_pts = all_pts
		self.all_names = all_names
		self.all_parts = all_parts
		return all_pts, all_names, all_parts

	def estimate_cow_center(self):
		"""
		Called by add_parts(...).

		Used to initialize:
			- all_center_pts: A list of all the center points of the body parts.
			- weighted_center: Calculated by summing the product of each center point and its confidence score, and dividing by the sum of the confidence scores.
			- center_pt_dif: A list of the distances between each center point and the weighted center.
		"""
		all_center_pts, prob = [], []
		for partlist in self.part_dict.values():
			for part in partlist:
				all_center_pts.append(part.pt_center)
				prob.append(part.score)
		all_center_pts = np.array(all_center_pts)
		prob = np.array(prob)
		weighted_center = np.sum([all_center_pts[i]*prob[i] for i in range(len(all_center_pts))], axis = 0) / np.sum(prob)
		self.all_center_pts = all_center_pts
		self.weighted_center = weighted_center
		self.center_pt_dif = np.sqrt(np.sum((all_center_pts - weighted_center)**2, axis = 1))

	def cow_parts_sort(self):
		"""
		Sorts the body parts by confidence scores in descending order, used in ravel(...).

		Used to update:
			- part_dict: A dictionary mapping part names to lists of bodypart objects.
			- part_idxes: A dictionary mapping part names to lists of their indices.
			- count: A dictionary mapping part names to their counts.
		"""
		cow_part_idx = []
		for i, partname in enumerate(self.partnames):
			if self.count[partname] > 1:
				# here compare the points for the same joint - He 
				scores = []
				for part in self.part_dict[partname]:
					# current version is only use the scmap confidence score only - He
					scores.append(part.score)
				order = np.argsort(scores)[::-1]
				self.part_dict[partname] = [self.part_dict[partname][x] for x in order]

	def find_all_limbs(self,config, prediction = False):
		limbs = []
		if prediction:
			# predict the body region - He
			pts, names = self.bodypart_prediction(config)
			# add the leg parts - He
			for i, name in enumerate(self.all_names):
				if name in config.leg_region_points:
					names.append(name)
					pts = np.concatenate((pts, self.all_pts[i].reshape(-1,2)), axis = 0)
		else:
			names, pts = self.all_names, self.all_pts

		for limb in config.body_limbs + config.leg_limbs:
			if limb[0] in names and limb[1] in names:
				idx1 = names.index(limb[0])
				idx2 = names.index(limb[1])
				limbs.append([pts[idx1], pts[idx2]])
		return np.array(limbs)

	def bodypart_prediction(self,config, predicted_pts_only = False):
		"""
		Adds the predicted parts for any undetected upper body part. If a detected part exists for shoulder, spine, 
		tailbase, bottomback, and bottomfront, the predicted part is compared to the detected part. If the distance 
		between the predicted part and detected part exceeds a 300px threshold and the angle between the vectors from 
		the Cow's center to these parts exceeds 60 degrees, the predicted part is used. Otherwise, the detected part is used.

		Parameters:
		predicted_pts_only (bool): A flag indicating whether to return only the predicted points. Defaults to False.

		Returns:
		res_pts (list): A list of the estimated positions of the body parts.
		bodypart_list_spine.copy() (list): A copy of the list of the names of the body parts.
		"""
		center = self.weighted_center
		predicted_body_part_pts, _ = estimate_body_parts_from_center(center,config)

		predicted_body_part_pts = np.array(predicted_body_part_pts).astype(int)
		if predicted_pts_only: return predicted_body_part_pts, config.body_region_points.copy()

		# compare the predicted pts and detect pts - He 
		res_pts = []

		for i, partname in enumerate(config.body_region_points):
			pt_pred = predicted_body_part_pts[i]
			if partname not in self.all_names:
				res_pts.append(pt_pred)
				continue
			pt = self.all_pts[self.all_names.index(partname)]
			# check if we need to compare - He
			if partname in config.body_region_points and partname not in config.head_region_points: 
				# compare - He 
				dis = compute_distance(pt, pt_pred)
				ang = compute_angle(pt-center, pt_pred-center)
				# If distance between predicted and detected points is greater than 300
				# and angle between vectors from the center to these points is greater than pi/3, use predicted point. 
				# Otherwise use detected point
				if dis > 300 and ang > np.pi/3:
					res_pts.append(pt_pred)
					continue
			res_pts.append(pt)

		return res_pts, config.body_region_points.copy()

def estimate_body_parts_from_center(center,config):    
	"""
	For every upper body part, a vector containing the respective bodyparts' average distance 
	from the center is added to the center point to estimate the position of the body part.
	
	is added to the center point to estimate the position of the body part.
	
	Used by cow.bodypart_prediction(...) and one_part_comparison(...).

    Parameters:
    center (ndarray): The center point of the cow.

    Returns:
    predicted_body_part_pts (List[ndarray]): A list of the estimated positions of the body parts.
    predicted_body_part_names (List[str]): A list of the names of the body parts.
    """
	
	predicted_body_part_pts = []
	predicted_body_part_names = []
	for partname in config.body_region_points:
		# center is in (row,col)/(y,x) format and trained_part_model_dict is in (x,y) format
		# center is temporarily flipped to (y,x) for addition
		vec = np.array(config.cow_centroid_model[partname])[0] + center[::-1] 
		# vec is flipped back to (x,y) format before appending to predicted_body_part_pts
		predicted_body_part_pts.append(vec[::-1])
		predicted_body_part_names.append(partname)
	return predicted_body_part_pts, predicted_body_part_names

def one_part_comparison(parts, center, config):
	"""
	Used to compare multiple detections of the same body part on the same cow to find the one of best fit.
	Called by eliminate_duplicates(...).
	
	Parameters:
	parts (List[bodypart]): A list of bodypart objects, which are all of the same kind on the same cow.
	center (ndarray): The center point of the cow.

	Returns:
	parts[np.argmin(angles)] (bodypart): The bodypart object with the smallest angle between the vector from the center to the body part and the vector from the center to the predicted body part.
	"""
	p_pts, p_names = estimate_body_parts_from_center(center,config)
	# all multiple parts in the same cow, select the best fit - He
	distances, angles = [], []
	for part in parts:
		pt = part.pt
		pt_pred = p_pts[p_names.index(part.name)]
		dis = compute_distance(pt, pt_pred)
		ang = compute_angle(pt-center, pt_pred-center)
		distances.append(dis)
		angles.append(ang)
	return parts[np.argmin(angles)]

def eliminate_duplicates(parts_list,config):
	center_list = [x.pt_center for x in parts_list]
	center = np.median(center_list, axis = 0)
	res_list = []
	for name in config.body_region_points:
		parts = [x for x in parts_list if x.name == name]
		if len(parts) == 0: continue
		res_list.append(one_part_comparison(parts, center,config))
	return res_list

def cows_selection(cows):
	new_cows = []
	for cow in cows:
		center_h = cow.weighted_center[0]
		if len(cow.all_pts) < 4:# body region should have 9
			continue
		new_cows.append(cow)
	return new_cows

def cow_points_cluster(parts, config):
	# Given one point and its part label, estimate the cow center based on trained model - He
	if len(parts) == 0: return []
	# project all bodypart points to center based on trained model - He 
	cow_center = []

	for part in parts:
		mu = np.array(config.cow_centroid_model[part.name][0])
		std = np.array(config.cow_centroid_model[part.name][1])
		part.pt_center = part.pt - mu[::-1]
		part.pt_center_std = std
		cow_center.append(part.pt_center)
	cow_center = np.array(cow_center).astype(int)

	# use Meanshift to cluster the number of cows
	clusters = MeanShift(bandwidth = 100).fit(cow_center).labels_
	
	# split the points onto each cow
	cows = []

	for i in np.unique(clusters):
		selected_parts = [parts[x] for x in range(len(parts)) if clusters[x] == i]
		selected_parts = eliminate_duplicates(selected_parts, config)
		cows.append(cow(config, selected_parts))
	
	cows = cows_selection(cows)
	cows = sorted(cows, key = lambda x:x.weighted_center[1])[::-1]

	return cows
