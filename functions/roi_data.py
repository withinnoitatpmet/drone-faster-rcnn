import numpy as np
import pdb
import math
from . import rpn_gt
import copy
import time

def calc_xy(A,regr,cols,rows,W,H,):
	
	X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

	A[0, :, :] = X - W/2
	A[1, :, :] = Y - H/2
	A[2, :, :] = W
	A[3, :, :] = H

	A = apply_regr_np(A,regr)
	A[2, :, :] = np.maximum(1, A[2, :, :])
	A[3, :, :] = np.maximum(1, A[3, :, :])
	A[2, :, :] += A[0, :, :]
	A[3, :, :] += A[1, :, :]

	A[0, :, :] = np.maximum(0, A[0, :, :])
	A[1, :, :] = np.maximum(0, A[1, :, :])
	A[2, :, :] = np.minimum(cols-1, A[2, :, :])
	A[3, :, :] = np.minimum(rows-1, A[3, :, :])

	return A

def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def apply_regr_np(X, T):
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

	if len(boxes) == 0:
		return []


	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)


	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs

def roi_input_data(rpn_class, rpn_regr, C, max_boxes=300,overlap_thresh=0.9):

	rpn_regr = rpn_regr / C.std_scaling

	sizes = C.anchor_box_scales
	ratios = C.anchor_box_ratios

	assert rpn_class.shape[0] == 1


	(rows, cols) = rpn_class.shape[1:3]

	anchor_count = 0

	tmp_boxes = np.zeros((4, rpn_class.shape[1], rpn_class.shape[2], rpn_class.shape[3]))


	for anchor_size in sizes:
		for anchor_ratio in ratios:

			anchor_w = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_h = (anchor_size * anchor_ratio[1])/C.rpn_stride

			regr = rpn_regr[0, :, :, 4 * anchor_count:4 * anchor_count + 4]
			regr = np.transpose(regr, (2, 0, 1))

			tmp_boxes[:, :, :, anchor_count] = calc_xy(tmp_boxes[:, :, :, anchor_count],regr,cols,rows,anchor_w,anchor_h)

			anchor_count += 1

	boxes = np.reshape(tmp_boxes.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
	probs = rpn_class.transpose((0, 3, 1, 2)).reshape((-1))

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	boxes = np.delete(boxes, idxs, 0)
	probs = np.delete(probs, idxs, 0)

	result = non_max_suppression_fast(boxes, probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result
	


def calc_iou(R, img_data, C, class2int):

	bb = img_data['bb']
	(width, height) = (img_data['width'], img_data['height'])
	# get image dimensions for resizing
	(resized_width, resized_height) = rpn_gt.get_new_img_size(width, height, C.im_size)

	gt_anchors = np.zeros((len(bb), 4))

	for bbox_num, bbox in enumerate(bb):
		gt_anchors[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
		gt_anchors[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
		gt_anchors[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
		gt_anchors[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = [] 

	for ix in range(R.shape[0]):
		(x1, y1, x2, y2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bb)):
			curr_iou = rpn_gt.iou([gt_anchors[bbox_num, 0], gt_anchors[bbox_num, 2], gt_anchors[bbox_num, 1], gt_anchors[bbox_num, 3]], [x1, y1, x2, y2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			x_roi.append([x1, y1, w, h])
			IoUs.append(best_iou)

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bb[best_bbox]['class']
				cxg = (gt_anchors[best_bbox, 0] + gt_anchors[best_bbox, 1]) / 2.0
				cyg = (gt_anchors[best_bbox, 2] + gt_anchors[best_bbox, 3]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gt_anchors[best_bbox, 1] - gt_anchors[best_bbox, 0]) / float(w))
				th = np.log((gt_anchors[best_bbox, 3] - gt_anchors[best_bbox, 2]) / float(h))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		class_num = class2int[cls_name]
		class_label = len(class2int) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		coords = [0] * 4 * (len(class2int) - 1)
		labels = [0] * 4 * (len(class2int) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx, sy, sw, sh = C.classifier_regr_std
			coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
			labels[label_pos:4+label_pos] = [1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	if len(x_roi) == 0:
		return None, None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
