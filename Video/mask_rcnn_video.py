
# python Video\mask_rcnn_video.py --input "Video\videos\video_2.mp4" --output "Video\output\myvideo_2.avi" --mask-rcnn "Video\mask-rcnn-coco"

import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([args["mask_rcnn"],"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

weightsPath = os.path.sep.join([args["mask_rcnn"],"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],"mask_rcnn_inception_v2_coco_2018_01_28.txt"])

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

vs = cv2.VideoCapture(args["input"])
writer = None

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	total = -1

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	(boxes, masks) = net.forward(["detection_out_final","detection_masks"])
	end = time.time()

	# loop over the number of detected objects
	for i in range(0, boxes.shape[2]):
		
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		if confidence > args["confidence"]:
			
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
			mask = (mask > args["threshold"])

			roi = frame[startY:endY, startX:endX][mask]

			color = COLORS[classID]
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			frame[startY:endY, startX:endX][mask] = blended

			color = [int(c) for c in color]
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color, 2)

			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(frame, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
vs.release()
