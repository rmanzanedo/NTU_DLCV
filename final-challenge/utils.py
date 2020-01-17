import os
import json
import torch
import scipy.misc
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
from itertools import groupby
import json
import glob

import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

import parser

def equalizeHist(img):
	img_y_cr_cb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
	y, cr, cb = cv.split(img_y_cr_cb)
	# Applying equalize Hist operation on Y channel.
	y_eq = cv.equalizeHist(y)
	img_y_cr_cb_eq = cv.merge((y_eq, cr, cb))
	img_rgb_eq = cv.cvtColor(img_y_cr_cb_eq, cv.COLOR_YCR_CB2BGR)

	return img_rgb_eq

def checkColor(patch):
	hsv_patch = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
	orange_low = np.array([5, 50, 50])
	orange_high = np.array([20, 255, 255])
	orange_mask = cv.inRange(hsv_patch, orange_low, orange_high)
	#print(orange_mask)
	#print([orange_mask==255][0].sum())
	if([orange_mask==255][0].sum()>=300):
		return True
	else:
		return False

def checkKeyPointsPos(patch_coord, kp_coordinates, patch_size, difficulty='easy'):
	n_kps = len(kp_coordinates)
	count = 0
	for i in range(n_kps):
		kp_x = kp_coordinates[i][0]
		kp_y = kp_coordinates[i][1]
		patch_x = patch_coord[0]
		patch_y = patch_coord[1]
		if(kp_x>=patch_x and kp_x<patch_x+patch_size and kp_y>=patch_y and kp_y<patch_y+patch_size):
			count += 1
	if(difficulty == 'easy'):
		thr = 100
	elif(difficulty == 'hard'):
		thr = 50
	#print('int(n_kps/thr): {}'.format(int(n_kps/thr)))
	if(count>=int(n_kps/thr)):
		return True
	else:
		return False


def getValidPatches(img, reshape_shape, patch_size):
	#Define feature detector
	orb = cv.ORB_create()
	#Blur the image
	width = img.shape[0]
	height = img.shape[1]
	equalized_img = equalizeHist(img.copy())
	#viewImage(equalized_img, 'Equilized image', destroy=False)
	blurred_img = cv.GaussianBlur(equalized_img, (7, 7), 0)
	#viewImage(blurred_img, 'Blurred image', destroy=False)
	#Get keypoints from ORB
	kp, des = orb.detectAndCompute(blurred_img, None)
	img_kp = cv.drawKeypoints(blurred_img.copy(), kp, None, color=(0,255,0), flags=0)
	#viewImage(img_kp, 'Image with keypoints', destroy=False)
	kp_coordinates = [kp[i].pt for i in range(len(kp))]
	#print('kp_coordinates: {}'.format(kp_coordinates))
	#Get patches from original image
	patches = []
	for i in range(0,height,patch_size):
		for j in range(0,width,patch_size):
			patch = blurred_img[j:j+patch_size, i:i+patch_size, :]
			patches.append([patch, (i,j)])
	#Get valid patches
	valid_patches = []
	for i in range(len(patches)):
		patch = patches[i][0]
		patch_coord = patches[i][1]
		#viewImage(patch, 'Patch image', destroy=False)
		color_condition = checkColor(patch)
		kp_condition = checkKeyPointsPos(patch_coord, kp_coordinates, patch_size)
		if(color_condition and kp_condition):
			valid_patches.append(patches[i]) 
		#print(patches[i][1])
	cv.destroyAllWindows()
	if(len(valid_patches) == 0):
		for i in range(len(patches)):
			patch_coord = patches[i][1]
			kp_condition = checkKeyPointsPos(patch_coord, kp_coordinates, patch_size, difficulty='hard')
			if(kp_condition):
				valid_patches.append(patches[i]) 

	#for i in range(len(valid_patches)):
		#print('valid patch coordinates: {}'.format(valid_patches[i][1]))
		#viewImage(valid_patches[i][0], 'Valid patch image', destroy=False)
	#cv.destroyAllWindows()

	return valid_patches


def getBoundingBox(original_img, valid_patches, patch_size):
	sum_x = 0
	sum_y = 0
	for valid_patch in valid_patches:
		valid_patch_coord = valid_patch[1]
		sum_x = sum_x + valid_patch_coord[0]+patch_size/2
		sum_y = sum_y + valid_patch_coord[1]+patch_size/2
	baricenter = (int(sum_x/len(valid_patches)), int(sum_y/len(valid_patches)))
	#print('baricenter: {}'.format(baricenter))
	image = cv.circle(original_img.copy(), baricenter, radius = 5, color = (255,0,0), thickness = 2)

	if(baricenter[1]-50 >= 0):
		start_x = baricenter[1]-50
	else:
		start_x = 0
	if(baricenter[1]+50 < original_img.shape[0]):
		stop_x = baricenter[1]+50
	else:
		stop_x = original_img.shape[0]
	if(baricenter[0]-128 >= 0):
		start_y = baricenter[0]-128
	else:
		start_y = 0
	if(baricenter[0]+128 < original_img.shape[1]):
		stop_y = baricenter[0]+128
	else:
		stop_y = original_img.shape[1]
	final_img = original_img[start_x:stop_x, start_y: stop_y, :]
	args = parser.arg_parse()
	final_img = cv.resize(final_img, (int(args.img_shape[0]/2), int(args.img_shape[1]/2)), cv.INTER_CUBIC)
	#print(final_img.shape)
	return final_img

def getLocalImage(img_path, reshape_shape, patch_size):
	original_img = cv.imread(img_path)
	original_img = cv.resize(original_img, reshape_shape, cv.INTER_CUBIC)
	box_img = original_img.copy()
	valid_patches = getValidPatches(original_img, reshape_shape, patch_size)
	final_img = getBoundingBox(original_img, valid_patches, patch_size)
	return final_img