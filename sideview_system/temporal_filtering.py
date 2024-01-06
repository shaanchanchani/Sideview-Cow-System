import imageio
import scipy.stats
import numpy as np
import os, glob, copy
from .config import Config
from .utils import compute_angle, compute_distance, pt_in_region, line_angle, list_reshape, compute_distance_lists
from .drawing import draw_pp_on_video, draw_post_processing
from .cow_clustering import cow
from .body_region_part_extraction import bodypart
	
def temporal_lpf_2d(x):
	l = len(x)
	new_x = np.ones(x.shape) * -1
	for i in range(2, l - 2):
		win = x[i-2:i+3, :]
		idx = [win[:,0] != -1]
		win = win[idx]
		if len(win) != 0:
			new_x[i,:] = np.median(win, axis = 0)
	return new_x

def temporal_lpf(y):
	l = len(y)
	new_y = y.copy()
	new_y[2:l-2] = -1
	for i in range(2, l - 2):
		win = y[i-2:i+3]
		win = win[win != -1]
		if len(win) != 0:
			new_value = np.median(win)
			if new_value != -1:
				new_y[i] = new_value
	return new_y

# this function gives each cow a unique label in the video
# so that each cow has its own color in one video
def temporal_label_cows(cows_list,config):
	MAX_cow_id = 0
	cows_id_list = []
	cows_track_dic = {}
	cows_last_frame_dic = {}
	
	for i, cows in enumerate(cows_list):
		if len(cows) == 0: 
			cows_id_list.append([])
			continue

		# search new
		cur_ids = []
		for cow_idx, cow in enumerate(cows):
			# for each cow in the new frame
			cur_cow_body_pts = np.array(cow.bodypart_prediction(config)[0])
			distance_list = []
			# search the existing cow dic, try to find match
			for temp_cow_idx in cows_track_dic:
				# check if this track should be forget
				if i - cows_last_frame_dic[temp_cow_idx] > 50:
					distance_list.append(99999)
				else:
					labeled_cow_pts = cows_track_dic[temp_cow_idx].copy()
					distance = compute_distance_lists(cur_cow_body_pts, labeled_cow_pts)
					distance_list.append(np.min(distance))

			# after computed distance, find if matched to storage
			if len(distance_list) != 0 and np.min(distance_list) < 50:
				# if founded
				matched_id = np.argmin(distance_list)
				cows_track_dic[matched_id] = cur_cow_body_pts.copy()
				cows_last_frame_dic[matched_id] = i
				# save this id to the cur_ids
				cur_ids.append(matched_id)
			else:
				# if now, add a new one to the dic
				new_cow_idx = len(cows_track_dic)
				cows_track_dic[new_cow_idx] = cur_cow_body_pts.copy()
				cows_last_frame_dic[new_cow_idx] = i
				cur_ids.append(new_cow_idx)
		# save current cows id to the final list
		cows_id_list.append(cur_ids)

	# recover the cows_list, with each row represent one unique cow
	max_cow_n = len(cows_track_dic)
	new_cows_list = []
	for i in range(len(cows_id_list)):
		cur_cows = [[] for _ in range(max_cow_n)]
		for cow_idx, cow in enumerate(cows_list[i]):
			cur_cows[cows_id_list[i][cow_idx]] = cow
		new_cows_list.append(cur_cows)
	# new_cows_list = np.array(new_cows_list)

	final_cow_list = []
	for cow_idx in range(max_cow_n):
		temp_cows_list = [x[cow_idx] for x in new_cows_list]
		length_count = np.array([len(x) for x in temp_cows_list])
		if np.sum(length_count != 0) > 20:
			final_cow_list.append(temp_cows_list)
	if len(final_cow_list) == 0:
		final_cow_list = new_cows_list[:]
	else:
		final_cow_list = list_reshape(final_cow_list)
	return final_cow_list


def temporal_pp(cows_list, config, filter_part_list = []):
	filter_part_list = [x for x in config.body_region_points if x not in config.head_region_points]
	l = len(cows_list)
	if l == 0: return []
	max_cow_n = np.max([len(x) for x in cows_list])
	temp = -1*np.ones((l, max_cow_n, 2))
	# extraction from cows_list
	pts_traj = {}
	for partname in config.body_region_points:
		pts_traj[partname] = temp.copy()
	for i in range(l):
		for j in range(len(cows_list[i])):
			for partname in config.body_region_points:
				cur_cow = cows_list[i][j]
				if len(cur_cow) == 0:
					continue
				if partname in cur_cow.all_names:
					idx = cur_cow.all_names.index(partname)
					pts_traj[partname][i,j,:] = cur_cow.all_pts[idx]
	pts_traj_orig = copy.copy(pts_traj)
	# lpf
	for j in range(max_cow_n):
		for k, partname in enumerate(filter_part_list):
			orig_pts = pts_traj[partname][:,j,:]
			pts = orig_pts.copy()
			pts_traj[partname][:,j,0] = temporal_lpf(pts[:,0])
			pts_traj[partname][:,j,1] = temporal_lpf(pts[:,1])

	new_cows_list = cows_list[:]

	for i in range(len(cows_list)):
		for j in range(max_cow_n):
			for k, partname in enumerate(filter_part_list):
				pt = pts_traj[partname][i,j,:]
				if len(new_cows_list[i][j]) == 0: continue
				if np.sum(pt) > 0:
					if partname not in new_cows_list[i][j].all_names:
						new_cows_list[i][j].all_pts.append(pt)
						new_cows_list[i][j].all_names.append(partname)
						new_cows_list[i][j].all_parts.append(bodypart(name = partname, pt = pt, score = 1))
					else:
						idx = new_cows_list[i][j].all_names.index(partname)
						new_cows_list[i][j].all_pts[idx] = pt
						new_cows_list[i][j].all_parts[idx] = bodypart(name = partname, pt = pt, score = 1)
	return new_cows_list
