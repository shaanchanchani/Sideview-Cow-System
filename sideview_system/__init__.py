import glob
import numpy as np
import os 
import imageio.v2 as imageio
import tensorflow as tf
import pickle
from .config import trained_models_dict, Config
from .body_region_part_extraction import extract_body_region_parts, group_by_distance, merge_bodyparts
from .utils import generate_dif_video, write_to_video
from .drawing import draw_pp_on_video, draw_post_processing
from .cow_clustering import cow_points_cluster 
from .leg_region_part_extraction import leg_detection_from_scmap, fit_leg_points_to_cows
from .temporal_filtering import temporal_label_cows, temporal_pp
from deeplabcut.pose_estimation_tensorflow.core.predict import setup_pose_prediction, extract_cnn_output
from deeplabcut.pose_estimation_tensorflow.config import load_config

def init_deeplabcut(model_info):
	print('Loading DLC cfg file:')
	print(model_info['folder'] + 'config.yaml')
	dlc_cfg = load_config(model_info['folder'] + model_info['dlc_cfg'])
	dlc_cfg['init_weights'] = model_info['folder'] + model_info['init_weights']
	dlc_cfg['batch_size'] = 1
	tf.compat.v1.reset_default_graph()
	sess, inputs, outputs = setup_pose_prediction(dlc_cfg)
	return sess, inputs, outputs, dlc_cfg

def generate_dcnn_output(img_dir, net_para = []):
	if type(img_dir) == str and os.path.exists(img_dir):
		image = imageio.imread(img_dir)
	else:
		image = img_dir.copy()
	SESS, INPUTS, OUTPUTS, dlc_cfg = net_para
	im = np.expand_dims(image, axis=0).astype(float)
	outputs_np = SESS.run(OUTPUTS, feed_dict={INPUTS: im})
	scmap, locref = extract_cnn_output(outputs_np, dlc_cfg)
	return scmap

def run_on_video(video_dir, config, output_dir = '', net_para = []):
	reader = imageio.get_reader(video_dir)
	scmap_list, parts_list, cows_list = [], [], []
	for idx, frame in enumerate(reader):
		img_idx = str(idx)
		scmap = generate_dcnn_output(frame, net_para = net_para)
		parts = extract_body_region_parts(scmap,config,net_para[3])
		cows = cow_points_cluster(parts,config)
		cows, leg_parts = leg_detection_from_scmap(cows, scmap, config, net_para[3])
		scmap_list.append(scmap)
		parts_list.append(parts + leg_parts)
		cows_list.append(cows)
	cows_list = temporal_label_cows(cows_list,config)
	cows_list = temporal_pp(cows_list, config, filter_part_list = config.body_region_points)
	if len(output_dir) != 0:
		write_to_video(video_dir, cows_list, config, output_dir = output_dir)
	return [scmap_list, parts_list, cows_list]

def pp_parts_from_two_nets_npy(video_name, config, folder1, folder2):
	# npy1_dir_dif_list = glob.glob(folder1 + '*' + video_name + '_output.npy')
	npy1_dir_dif_list = glob.glob(folder1 + '*' + video_name + '_output.p')
	# npy2_dir_color_list = glob.glob(folder2 + '*' + video_name + '_output.npy')
	npy2_dir_color_list = glob.glob(folder2 + '*' + video_name + '_output.p')
	assert len(npy1_dir_dif_list) == 1 and len(npy2_dir_color_list) == 1, 'this video name is not unique: ' + video_name 
	npy1_dir_dif, npy2_dir_color = npy1_dir_dif_list[0], npy2_dir_color_list[0]
	# part_list1 = np.load(npy1_dir_dif, allow_pickle=True)
	part_list1 = pickle.load(open(npy1_dir_dif, 'rb'))

	#Removes first frame for difference model because it doesn't have a previous frame to compare w/
	part_list1 = part_list1[1:]
	# part_list2 = np.load(npy2_dir_color, allow_pickle=True)
	part_list2 = pickle.load(open(npy2_dir_color, 'rb'))

	merged_partlist, merged_cowlist, = [], []
	for i in range(len(part_list1)):
		all_parts = part_list1[i] + part_list2[i]
		body_parts, leg_parts = [], []

		for partname in config.body_region_points:
			some_parts = [x for x in all_parts if x.name == partname]
			pt_candidate_list = group_by_distance(some_parts, th = 50)
			for idxes in pt_candidate_list:
				body_parts.append(merge_bodyparts([some_parts[x] for x in idxes]))

		for partname in config.leg_region_points:
			some_parts = [x for x in part_list2[i] if x.name == partname]
			if len(some_parts) == 0:
				some_parts = [x for x in part_list1[i] if x.name == partname]
			some_parts = sorted(some_parts, key = lambda x:x.score)[::-1]
			leg_parts.extend([x for x in some_parts])

		cows = cow_points_cluster(body_parts,config)
		new_cows, new_leg_parts = fit_leg_points_to_cows(cows, leg_parts, config)
		merged_partlist.append(body_parts + new_leg_parts)
		merged_cowlist.append(new_cows)
	return merged_partlist, merged_cowlist

def runner(trained_models, input_video_folder, output_folder, config, dif_folder = ''):
	if type(input_video_folder) == str:
		input_video_dirs = glob.glob(input_video_folder + '*.avi')
	elif type(input_video_folder) == list:
		input_video_dirs = input_video_folder[:]
	else:
		raise IOError('unknown input_video_folder ')

	print('Loading Color CNN')
	SESS, INPUTS, OUTPUTS, dlc_cfg = init_deeplabcut(trained_models['color_model'])

	print('Running on color videos')
	orig_res_dir = output_folder + 'orig_res/'
	if not os.path.exists(orig_res_dir):
		os.makedirs(orig_res_dir, exist_ok=True)
	for i, video_dir in enumerate(input_video_dirs):
		if os.path.exists(orig_res_dir + os.path.basename(video_dir)[:-4] + '_output.p'): continue
		print(i, len(input_video_dirs))
		scmaps, parts, cows_list = run_on_video(video_dir, config, net_para = [SESS, INPUTS, OUTPUTS, dlc_cfg])
		# np.save(orig_res_dir + os.path.basename(video_dir)[:-4] + '_output.npy', np.array(parts))
		pickle.dump(parts, open(orig_res_dir + os.path.basename(video_dir)[:-4] + '_output.p', 'wb'))

	print('Loading Frame Difference CNN')
	SESS, INPUTS, OUTPUTS, dlc_cfg = init_deeplabcut(trained_models['dif_model'])
	dif_res_dir = output_folder + 'dif_res/'
	if not os.path.exists(dif_res_dir):
		os.makedirs(dif_res_dir, exist_ok=True)
	for i, video_dir in enumerate(input_video_dirs):
		print(i, os.path.basename(video_dir), len(input_video_dirs))
		if os.path.exists(dif_folder + os.path.basename(video_dir)[:-4] + '.avi'):
			dif_video_path = dif_folder + os.path.basename(video_dir)[:-4] + '.avi'
			print('Found Frame Differencee Video')
		else:
			dif_video_path = output_folder + 'dif_videos/' + os.path.basename(video_dir)[:-4] + '.avi'
			if not os.path.exists(dif_video_path):
				if not os.path.exists(output_folder + 'dif_videos/'): 
					os.makedirs(output_folder + 'dif_videos/')
				print('Frame Difference video not found, generating...')
				generate_dif_video(video_dir, output_dir = dif_video_path)
			else:
				print('Found Frame Difference Video')
		if os.path.exists(dif_res_dir + os.path.basename(video_dir)[:-4] + '_output.p'): continue
		scmaps, parts, cows_list = run_on_video(dif_video_path, config, net_para = [SESS, INPUTS, OUTPUTS, dlc_cfg])
		# np.save(dif_res_dir + os.path.basename(dif_video_path)[:-4] + '_output.npy', np.array(parts))
		pickle.dump(parts, open(dif_res_dir + os.path.basename(dif_video_path)[:-4] + '_output.p', 'wb'))

	print('Merging results from two CNNs')
	merge_res_dir = output_folder + 'merge_res/'
	if not os.path.exists(merge_res_dir):
		os.makedirs(merge_res_dir, exist_ok=True)
    
	for video_dir in input_video_dirs:
		print(video_dir)
		video_name = os.path.basename(video_dir)[:-4]
		if os.path.exists(merge_res_dir + video_name + '_output.p'):
			continue
		parts_list, cows_list = pp_parts_from_two_nets_npy(video_name, config, dif_res_dir, orig_res_dir)
		cows_list = temporal_label_cows(cows_list, config)
		cows_list = temporal_pp(cows_list, config, filter_part_list = config.body_region_points)
		draw_pp_on_video(video_dir, cows_list, config, save_dir = merge_res_dir + video_name + '.avi', draw_predicted_pts = False)
		pickle.dump(cows_list, open(merge_res_dir + video_name + '_output.p', 'wb'))

def run_sideview_system(input_video_folders, output_folder, cow_model_key, dif_folder = 'dif_videos/'):
	"""
    Runs the sideview analysis system on videos in each specified folder using a given model.

    Args:
        input_video_folders (list of str): A list of folder names, each containing test video files. These folders should be in the same directory as the library.
        cow_model_key (str): The name of the model to be used for analysis.
        output_folder (str): The directory where analysis results will be saved.
        dif_folder(str, optional): The directory for saving difference videos. Defaults to 'dif_videos/'.

    """
	config = Config(cow_model_key)
	for folder_name in input_video_folders:
		save_folder = output_folder + cow_model_key + '_on_' + folder_name + '/'
		parent_dir = os.path.dirname(save_folder)
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir, exist_ok=True)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder, exist_ok=True)
		print(cow_model_key + '_on_' + folder_name +'\n')
		runner(trained_models_dict[cow_model_key], f'./{folder_name}/', save_folder, config, dif_folder = dif_folder)

