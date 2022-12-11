# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import pickle
import time
import cv2
import os
import psutil
import threading

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output fps plot")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

prev_time = 0
cpu_usage = 0
mem_usage = ""
fps_record = []

ended = False

def loop():
	while True:
		if ended:
			return
		
		global cpu_usage
		global mem_usage
		cpu_usage = psutil.cpu_percent(interval=1)
		phymem = psutil.virtual_memory()
		mem_usage = "RAM %5s%% %6s/%s"%(
            phymem.percent,
            str(int(phymem.used/1024/1024))+"M",
            str(int(phymem.total/1024/1024))+"M"
            )
		time.sleep(1)

t = threading.Thread(target=loop)
t.start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			if startX > endX:
				t = startX
				startX = endX
				endX = t
				t = startY
				startY = endY
				endY = t

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]

			try:
				face = cv2.resize(face, (32, 32))
			except Exception as e:
				print("ERROR: ", startY, startX, endY, endX)
				continue

			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# draw the label and bounding box on the frame
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	cur_time = time.time()
	fps = 1 / (cur_time - prev_time)
	prev_time = cur_time
	if fps > 0:
		fps_record.append(fps)

	cv2.putText(frame, "FPS {:.2f}".format(fps), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.putText(frame, "CPU {}%".format(cpu_usage), (10, 40),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	cv2.putText(frame, "{}".format(mem_usage), (10, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Liveness Detector", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		ended = True
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# plot the fps record
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(fps_record)), fps_record, label="fps")
plt.plot(np.arange(0, len(fps_record)), [sum(fps_record)/(max(len(fps_record),1))]*len(fps_record), label="avg")
plt.title("Fps Record")
plt.xlabel("Time")
plt.ylabel("FPS")
plt.legend(loc="lower left")
plt.savefig(args["plot"])