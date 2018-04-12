1.required python module & framework
h5py
opencv
numpy
keras
sklearn
tensorflow

2.dataset and model

standford drone dataset:http://cvgl.stanford.edu/projects/uav_data/

train and test data:
train:https://drive.google.com/open?id=1HUoRot-4Co_boeC_Yl9B3k5_fsfid_kJ
test:https://drive.google.com/open?id=1ulXhRmG9T2_coypw-8DqUH4uDze05zpH

model:https://drive.google.com/open?id=1svTOGxUU5zj4YcAHSef4fyUKW_u0s_iG

3.data generation

generator.py in datageneration folder is used to capture image from video and generating ground truth annotations

4.trainning 

in training stage,you need to specify folder path to data.txt,which can be found in train folder
each line contains `filename,x1,y1,x2,y2,class_name`,where x1,y1,x2,y2 is the groun truth bounding box

note that data.txt and training images should be in same folder

pre trained model is optional in this stage

folderpath  = 'folderpath'

5.testing
 
in this stage, you are required to download the trained model to repulicate our results and path is specified in config.py

img_path = 'image_path' image path should be specified

image results can be founded in results_imgs folder and bounding box can be founded in haiya





