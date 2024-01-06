import numpy as np
import copy
from .utils import line_angle, pt_in_region
from .cow_clustering import cow
from .body_region_part_extraction import get_candidate_parts_from_cmap_channel, bodypart

# extract leg points from scmaps
def search_leg_points(multi_channel_cmap, region, config, dlc_cfg):
	all_point_names = [x.lower() for x in dlc_cfg["all_joints_names"]]
	leg_parts = []
	for i, partname in enumerate(all_point_names):
		if partname in config.leg_region_points:
			single_channel_cmap = multi_channel_cmap[:,:,i]
			if partname == 'rightbackleg':
				parts = get_candidate_parts_from_cmap_channel(single_channel_cmap, 'rightbackhoof', th = 0.3)
			elif partname == 'rightbackhoof':
				parts = get_candidate_parts_from_cmap_channel(single_channel_cmap, 'rightbackleg', th = 0.3)
			else:
				parts = get_candidate_parts_from_cmap_channel(single_channel_cmap, partname, th = 0.3)
			for part in parts:
				if pt_in_region(part.pt, region):
					leg_parts.append(part)
	return leg_parts

def cow_add_legs(cowbody, config, new_parts = []):
	cow = copy.copy(cowbody)
	for partname in config.body_region_points + config.leg_region_points:
		if partname not in cow.partnames:
			cow.part_dict[partname] = []
			cow.part_idxes[partname] = []
			cow.count[partname] = 0
	cow.partnames = config.body_region_points + config.leg_region_points
	cow.add_parts(new_parts, compute_center = False)
	cow.ravel()
	return cow

def predict_leg_region(cow,config):
	merge_pts, merge_labels = cow.bodypart_prediction(config)
	pt1 = merge_pts[merge_labels.index('bottomfront')]
	pt2 = merge_pts[merge_labels.index('bottomback')]

	length_half = abs((pt2[1] - pt1[1])*3/4)
	length_mid = (pt2[1] + pt1[1])/2
	w1 = length_mid - length_half
	w2 = length_mid + length_half
	h1 = (pt2[0] + pt1[0])/2
	h2 = h1 + 200 # fixed height range
	return [w1,w2,h1,h2]

def leg_detection_from_scmap(cows, scmap, config, dlc_cfg, img = []): # legs main function
	full_cows, leg_parts_list = [], []
	for cow in cows:
		region = predict_leg_region(cow,config)
		leg_parts = search_leg_points(scmap, region, config, dlc_cfg)
		full_cow, leg_parts = fit_leg_points_to_cows([cow], leg_parts, config)
		full_cows.append(full_cow[0])
		leg_parts_list.extend(leg_parts)
	return full_cows, leg_parts_list

# leg merging and select leg points
def validate_leg_limb(part1, part2_list, region, angle_d = 70):
	# this angle r is the angle with arctan2(100, d)
	angle_range = [np.arctan2(100, -angle_d), np.arctan2(100, angle_d)]
	if part1 == 0: return 0
	if len(part2_list) == 0: return 0
	# check if it is inside region (belong to this cow)
	selected_leg_parts = []
	for leg_part in part2_list:
		if pt_in_region(leg_part.pt, region):
			selected_leg_parts.append(leg_part)
	if len(selected_leg_parts) == 0: return 0
	# point analysis
	ang = np.array([line_angle(part1.pt, x.pt) for x in selected_leg_parts])
	within_ang_idx = np.logical_and(ang < angle_range[0], ang > angle_range[1])# np.arctan2(100, -70) and arctan2(100, 70)
	res_leg_parts_list = [selected_leg_parts[x] for x in range(len(within_ang_idx)) if within_ang_idx[x]]

	if len(res_leg_parts_list) == 0:
		return 0
	else:
		return res_leg_parts_list[0]

# select leg and hoof pts and then add to cows
def fit_leg_points_to_cows(cows, leg_parts, config): 
	# convert leg_parts to leg_part_dic
	leg_part_dic = {x:[] for x in config.leg_region_points}
	for leg_part in leg_parts:
		leg_part_dic[leg_part.name].append(leg_part)

	# fit leg parts to cows
	full_cows, res_leg_parts_list = [], []
	for cow in cows:
		cow_legs_candidate = []
		# locate the leg regions
		region = predict_leg_region(cow,config)
		# find the original or estimated shoulder and tailbase point
		pts, labels = cow.bodypart_prediction(config)
		leg_part_dic['shoulder'] = [bodypart(name = 'shoulder', pt = pts[labels.index('shoulder')])]
		leg_part_dic['tailbase'] = [bodypart(name = 'tailbase', pt = pts[labels.index('tailbase')])]

		for i in range(len(config.leg_limbs))[::2]:
			leg_name1, leg_name2 = config.leg_limbs[i]
			leg_name3 = config.leg_limbs[i+1][1]
			# print(leg_name1, leg_name2, leg_name3)
			part2, part3 = 0, 0 
			# search shoulder to leg limb
			part1 = leg_part_dic[leg_name1][0]
			part2 = validate_leg_limb(part1, leg_part_dic[leg_name2], region)
			# if find leg joint, then use leg joint to find hoof
			if part2 != 0: 
				cow_legs_candidate.append(part2)
				part3 = validate_leg_limb(part2, leg_part_dic[leg_name3], region, angle_d = 100)
			# one more chance if leg is missing or leg cannot find hoof,
			# directly use shoulder to search hoof points
			if part3 == 0: 
				part3 = validate_leg_limb(part1, leg_part_dic[leg_name3], region)
			if part3 != 0: 
				cow_legs_candidate.append(part3)

		full_cow = cow_add_legs(cow, config, cow_legs_candidate)
		full_cows.append(full_cow)
		res_leg_parts_list.extend(cow_legs_candidate)
	return full_cows, res_leg_parts_list
