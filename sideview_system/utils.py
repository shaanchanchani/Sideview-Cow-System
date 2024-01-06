import numpy as np 
import cv2
import math 
import matplotlib.pyplot as plt
import copy 
from numpy.linalg import norm
# from .config import *
from .cow_clustering import cow
from skimage.color import rgb2gray
import os 
import imageio.v2 as imageio

def generate_dif_video(video_dir, output_dir = ''):
	if os.path.exists(output_dir):
		return
	reader = imageio.get_reader(video_dir)
	fps = reader.get_meta_data()['fps']
	writer = imageio.get_writer(output_dir)
	for i, frame in enumerate(reader):
		if i == 0:
			prev = frame.astype(float)/255.
		frame = frame.astype(float)/255.
		error_frame = np.abs(frame - prev)
		prev = frame.copy()

		error_frame = np.array(rgb2gray(error_frame) * 255, dtype = np.uint8)
		writer.append_data(error_frame)
		if i == 1:
			writer.append_data(error_frame)
	writer.close()

def write_to_video(video_dir, cows_list, config, output_dir = 'temp.avi'):
	reader = imageio.get_reader(video_dir)
	fps = reader.get_meta_data()['fps']
	writer = imageio.get_writer(output_dir, fps = fps)
	for idx, frame in enumerate(reader):
		frame = draw_post_processing(frame, cows_list[idx], config)
		writer.append_data(frame)
	writer.close()

def compute_distance(p0, p1):
	d = np.sqrt(np.sum((np.array(p0) - np.array(p1))**2))
	return d

def compute_distance_lists(list1, list2):
	res = []
	for i in range(len(list1)):
		res.append(compute_distance(list1[i], list2[i]))
	return np.array(res)

def compute_angle(l1, l2):
	# if two points are very close to each other
	if compute_distance(l1, l2) < 1: return 0
	return np.arccos(np.dot(l1, l2)/norm(l1)/norm(l2))

def merge_two_dict(dic1, dic2):
	dic3 = {}
	for key in dic1:
		dic3[key] = dic1[key] + dic2[key]
	return dic3

def pt_in_region(pt, region):
	w1,w2,h1,h2 = region
	return h1 < pt[0] < h2 and w1 < pt[1] < w2

def line_angle(a, b):
	d = np.array(b) - np.array(a)
	return np.arctan2(d[0], d[1])

def list_reshape(list_2d):
	if len(list_2d) == 0:
		return []
	if len(list_2d[0]) == 0:
		raise ValueError('?')
	m, n = len(list_2d), len(list_2d[0])
	new_list = []
	for i in range(n):
		new_list.append([list_2d[x][i] for x in range(m)])
	return new_list

def _c(ca,i,j,p,q):
	if ca[i,j] > -1:
		return ca[i,j]
	elif i == 0 and j == 0:
		ca[i,j] = np.linalg.norm(p[i]-q[j])
	elif i > 0 and j == 0:
		ca[i,j] = max( _c(ca,i-1,0,p,q), np.linalg.norm(p[i]-q[j]) )
	elif i == 0 and j > 0:
		ca[i,j] = max( _c(ca,0,j-1,p,q), np.linalg.norm(p[i]-q[j]) )
	elif i > 0 and j > 0:
		ca[i,j] = max(min(_c(ca,i-1,j,p,q), _c(ca,i-1,j-1,p,q), _c(ca,i,j-1,p,q)), np.linalg.norm(p[i]-q[j]))
	else:
		ca[i,j] = float('inf')
	return ca[i,j]


def frdist(p,q):
	""" 
	Computes the discrete Fréchet distance between
	two curves. The Fréchet distance between two curves in a
	metric space is a measure of the similarity between the curves.
	The discrete Fréchet distance may be used for approximately computing
	the Fréchet distance between two arbitrary curves, 
	as an alternative to using the exact Fréchet distance between a polygonal
	approximation of the curves or an approximation of this value.
	
	This is a Python 3.* implementation of the algorithm produced
	in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance. Tech. 
	Report CD-TR 94/64, Information Systems Department, Technical University 
	of Vienna.
	http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

	Function dF(P, Q): real;
		input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
		return: δdF (P, Q)
		ca : array [1..p, 1..q] of real;
		function c(i, j): real;
			begin
				if ca(i, j) > −1 then return ca(i, j)
				elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
				elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
				elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj ) }
				elsif i > 1 and j > 1 then ca(i, j) :=
				max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
				else ca(i, j) = ∞
				return ca(i, j);
			end; /* function c */

		begin
			for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
			return c(p, q);
		end.

	Parameters
	----------
	P : Input curve - two dimensional array of points
	Q : Input curve - two dimensional array of points

	Returns
	-------
	dist: float64
		The discrete Fréchet distance between curves `P` and `Q`.

	Examples
	--------
	>>> from frechetdist import frdist
	>>> P=[[1,1], [2,1], [2,2]]
	>>> Q=[[2,2], [0,1], [2,4]]
	>>> frdist(P,Q)
	>>> 2.0
	>>> P=[[1,1], [2,1], [2,2]]
	>>> Q=[[1,1], [2,1], [2,2]]
	>>> frdist(P,Q)
	>>> 0
	"""
	p = np.array(p, np.float64)
	q = np.array(q, np.float64)

	len_p = len(p)
	len_q = len(q)

	if len_p == 0 or len_q == 0:
		raise ValueError('Input curves are empty.')

	if len_p != len_q or len(p[0]) != len(q[0]):
		raise ValueError('Input curves do not have the same dimensions.')

	ca = (np.ones((len_p,len_q), dtype=np.float64) * -1 ) 

	dist = _c(ca,len_p-1,len_q-1,p,q)
	return dist