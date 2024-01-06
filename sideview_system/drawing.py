import numpy as np 
import cv2
import imageio.v2 as imageio

COLORS = [[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 0], [255, 255, 255], [0, 255, 255], [0, 0, 255], [0,0,0]]
np.random.seed(0)
COLORS = COLORS + [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(100)]

def draw_points(pts, frame = [], s = (720, 1280, 3), r = 1, c = (255, 0, 0), plot = True, copy = False, fill = -1):
	if len(frame) == 0:frame = np.zeros(s)
	if copy: frame = frame.copy()
	pts = np.array(pts).astype(int)
	for pt in pts:
		cv2.circle(frame, (pt[1], pt[0]), r, c, fill)
	if plot: plt.imshow(frame), plt.show()
	return frame

def draw_lines(pts, frame = [], s = (720, 1280, 3), w = 1, c = (255, 0, 0), plot = True, copy = False):
	if len(frame) == 0:frame = np.zeros(s)
	if copy: frame = frame.copy()
	pts = np.array(pts).astype(int)
	for pt in pts:
		cv2.line(frame,(pt[0][1], pt[0][0]),(pt[1][1], pt[1][0]),c,w)
	if plot: plt.imshow(frame), plt.show()
	return frame

def draw_crosses(pts, frame = [], s = (720, 1280, 3), r = 3, w = 1, c = (255, 0, 0), plot = True, copy = False):
	if len(frame) == 0:frame = np.zeros(s)
	if copy: frame = frame.copy()
	pts = np.array(pts).astype(int)
	for pt in pts:
		p0x, p0y = pt[1] - r, pt[0] - r
		p1x, p1y = pt[1] + r, pt[0] + r
		p2x, p2y = pt[1] - r, pt[0] + r
		p3x, p3y = pt[1] + r, pt[0] - r
		cv2.line(frame,(p0x, p0y),(p1x, p1y),c,w)
		cv2.line(frame,(p2x, p2y),(p3x, p3y),c,w)
	if plot: plt.imshow(frame), plt.show()
	return frame

def draw_post_processing(img, cows, config, pt_only = False, draw_predicted_pts = False):
	for i, cow in enumerate(cows):
		if len(cow) == 0: continue
		if not pt_only:
			# draw all possible limbs
			limbs = cow.find_all_limbs(config, prediction = True)
			img = draw_lines(np.array(limbs), img, w = 3, c = COLORS[i], plot = False)
		if draw_predicted_pts:
			# draw predicted body points
			img = draw_points(cow.bodypart_prediction(config)[0], img, r = 10, c = COLORS[i], plot = False, fill = 2)
			# draw cow center
			img = draw_crosses([cow.weighted_center], img, r = 15, w = 5, c = COLORS[i], plot = False)
		img = draw_points(cow.bodypart_prediction(config)[0], img, r = 10, c = COLORS[i], plot = False, fill = -1)
		# draw leg points
		leg_parts = [x.pt for x in cow.all_parts if x.name in config.leg_region_points]
		img = draw_points(np.array(leg_parts), img, r = 10, c = COLORS[i], plot = False)
	return img

def draw_pp_on_video(video_dir, cows_list, config, pt_only = False, draw_predicted_pts = False, save_dir = 'temp.avi'):
	reader = imageio.get_reader(video_dir)
	fps = reader.get_meta_data()['fps']
	writer = imageio.get_writer(save_dir, fps = fps)
	for i, frame in enumerate(reader):
		if i == len(cows_list): break
		img = draw_post_processing(frame, cows_list[i], config, pt_only)
		writer.append_data(img)
	writer.close()

def draw_pp_on_video_with_predicted_pts(video_dir, cows_list, config, save_dir = 'temp.avi'):
	reader = imageio.get_reader(video_dir)
	fps = reader.get_meta_data()['fps']
	writer = imageio.get_writer(save_dir, fps = fps)
	for i, frame in enumerate(reader):
		if i == len(cows_list): break
		img = draw_post_processing(frame, cows_list[i],config, draw_predicted_pts = True)
		writer.append_data(img)
	writer.close()