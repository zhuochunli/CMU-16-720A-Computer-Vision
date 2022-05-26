import numpy as np
import cv2

def loadVid(path):

	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name

	cap = cv2.VideoCapture(path)

	# Append frames to list
	frames = []

	# Check if camera opened successfully
	if cap.isOpened()== False:
	    print("Error opening video stream or file")

	# Read until video is completed
	while(cap.isOpened()):

	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    if ret:
	        #Store the resulting frame
	        frames.append(frame)
	    else:
	        break

	# When everything done, release the video capture object
	cap.release()
	frames = np.stack(frames)

	return frames
