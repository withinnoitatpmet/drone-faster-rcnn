import numpy as np
import cv2


#filepath,x1,y1,x2,y2,class_name

annotation = {}
#video_num = '2'
ratio = 1
startframe = 1
def get_data(input_path):
	
	with open(input_path,'r') as f:

		for line in f:
			line_split = line.strip().split(' ')
			(ID,x1,y1,x2,y2,frame,lost,occluded,generated,class_name) = line_split
			if int(frame)% ratio == 0 and int(frame)/ratio != 0 and int(frame) > startframe: 
				if lost == 1:
					pass
				else:
			
					with open('test\\test.txt', 'a') as the_nofile:
						the_nofile.write(str(int(int(frame)/ratio))+','+x1+','+y1+','+x2+','+y2+','+class_name+'\n')

		
	return 0
	

folder  =  'folderpath'
cap = cv2.VideoCapture(folder+'video.mov')
num_frame = 1


while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    if frame is None:
        break
    if num_frame%ratio ==0 and num_frame > startframe:
    	cv2.imwrite('video_1/'+str(int(num_frame/ratio))+'.jpg', frame)

    # cv2.imshow('image',frame)
    num_frame = num_frame+1

# When everything done, release the capture
print(num_frame)
cap.release()
cv2.destroyAllWindows()

#get_data(folder+'annotations.txt')