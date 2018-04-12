from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import cv2
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from functions import config, rpn_gt
from functions import losses as losses
import functions.roi_data as roi_data
from keras.utils import generic_utils
from functions import network_vgg as nn

folderpath  = 'E:\\stanford_campus_dataset\\data\\'
C = config.Config()
C.model_path = './model_frcnn.hdf5'
C.network = 'vgg'
C.base_net_weights = nn.get_weight_path()
config_output_filename = 'config.pickle'

sys.setrecursionlimit(40000)

def select_samples(Y):
	#class name is 'bg'
	neg = np.where(Y[0, :, -1] == 1)
	#class name is not 'bg'
	pos = np.where(Y[0, :, -1] == 0)


	if len(neg) > 0:
		neg = neg[0]
	else:
		neg = []

	if len(pos) >0:
		pos = pos[0]
	else:
		pos = []

	pos_num = len(pos)

	if pos_num < C.num_rois//2:
		sel_pos = pos.tolist()
	else:
		sel_pos = np.random.choice(pos, C.num_rois//2 ,replace = False).tolist()
	try:
		sel_neg = np.random.choice(neg, C.num_rois - len(sel_pos),replace = False).tolist()
	except:
		sel_neg = np.random.choice(neg, C.num_rois - len(sel_pos),replace = True).tolist()

	samples = sel_pos + sel_neg

	return samples, pos_num



def get_data(path):
	images = {}
	classes_count = {}
	class2int = {}
	
	with open(path,'r') as f:

		print('getting data')

		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,y1,x2,y2,class_name) = line_split
			filename = folderpath+filename


			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class2int:
				class2int[class_name] = len(class2int)

			if filename not in images:
				images[filename] = {}
				
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
				images[filename]['filepath'] = filename
				images[filename]['width'] = cols
				images[filename]['height'] = rows
				images[filename]['bb'] = []
				if np.random.randint(0,6) > 0:
					images[filename]['imageset'] = 'trainval'
				else:
					images[filename]['imageset'] = 'test'

			images[filename]['bb'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
		data = []
		for key in images:
			data.append(images[key])		
		return data, classes_count, class2int







dataset_imgs, all_classes, class2int = get_data(folderpath+'data.txt')

if 'bg' not in all_classes:
	all_classes['bg'] = 0
	class2int['bg'] = len(class2int)

C.class2int = class2int

print('Training images per class:')
pprint.pprint(all_classes)
print('Num classes = {}'.format(len(all_classes)))


with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)


random.shuffle(dataset_imgs)

train_input = [s for s in dataset_imgs if s['imageset'] == 'trainval']

data_gen_train = rpn_gt.get_anchor_gt(train_input, all_classes, C, nn.get_img_output_length, mode='train')

img_input_shape = Input(shape=(None,None,3))
roi_input_shape = Input(shape=(None, 4))

cnn_layers = nn.vgg(img_input_shape)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_network = nn.rpn(cnn_layers, num_anchors)

classifier_network = nn.classifier(cnn_layers, roi_input_shape, C.num_rois, nb_classes=len(all_classes))

rpn_model = Model(img_input_shape, rpn_network[:2])
classifier_model = Model([img_input_shape, roi_input_shape], classifier_network)

# model generation
model_combined = Model([img_input_shape, roi_input_shape], rpn_network[:2] + classifier_network)

try:
	print('loading pretrained model')
	rpn_model.load_weights(C.base_net_weights, by_name=True)
	classifier_model.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model')

optimizer = Adam(lr=1e-5)
rpn_model.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
classifier_model.compile(optimizer=optimizer, loss=[losses.class_loss_cls, losses.class_loss_regr(len(all_classes)-1)], metrics={'dense_class_{}'.format(len(all_classes)): 'accuracy'})
model_combined.compile(optimizer='sgd', loss='mae')

epoch_length = 1000
num_epochs = 2000
iter_num = 0

total_losses = np.zeros((epoch_length, 5))

start_time = time.time()

best_loss = np.Inf

print('Starting training')

for epoch_num in range(num_epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:

			X, Y, img_data = next(data_gen_train)

			loss_rpn = rpn_model.train_on_batch(X, Y)

			P_rpn = rpn_model.predict_on_batch(X)

			R = roi_data.roi_input_data(P_rpn[0], P_rpn[1], C, overlap_thresh=0.7, max_boxes=300)
			X2, Y1, Y2, IouS = roi_data.calc_iou(R, img_data, C, class2int)
			#sample

			samples, pos_num = select_samples(Y1)

			loss_class = classifier_model.train_on_batch([X, X2[:, samples, :]], [Y1[:, samples, :], Y2[:, samples, :]])

			total_losses[iter_num, 0] = loss_rpn[1]
			total_losses[iter_num, 1] = loss_rpn[2]

			total_losses[iter_num, 2] = loss_class[1]
			total_losses[iter_num, 3] = loss_class[2]
			total_losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(total_losses[:iter_num, 0])), ('rpn_regr', np.mean(total_losses[:iter_num, 1])),
									  ('detector_cls', np.mean(total_losses[:iter_num, 2])), ('detector_regr', np.mean(total_losses[:iter_num, 3]))])

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(total_losses[:, 0])
				loss_rpn_regr = np.mean(total_losses[:, 1])
				loss_class_cls = np.mean(total_losses[:, 2])
				loss_class_regr = np.mean(total_losses[:, 3])

				
				print('Loss RPN classifier: {}'.format(loss_rpn_cls))
				print('Loss RPN regression: {}'.format(loss_rpn_regr))
				print('Loss Detector classifier: {}'.format(loss_class_cls))
				print('Loss Detector regression: {}'.format(loss_class_regr))
				print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_combined.save_weights(C.model_path)

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete')
