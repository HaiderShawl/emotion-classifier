# importing the necessary libraries
import cv2
import numpy as np
import emotionClassifier

# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('pitch.mov')
# cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotionClassifier = emotionClassifier.emotionClassifier()

# Loop until the end of the video
while (cap.isOpened()):

	# Capture frame-by-frame
	ret, frame = cap.read()
	frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
						interpolation = cv2.INTER_CUBIC)

	# Display the resulting frame
	cv2.imshow('Frame', frame)

	# conversion of BGR to grayscale is necessary to apply this operation
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = gray[y:y+h, x:x+w]

		face = frame[y:y+h, x:x+w]
		face = cv2.resize(face, (48, 48),
               interpolation = cv2.INTER_NEAREST)



		emotionClassifier.predict(face[::-1])
		cv2.imshow('face', face)



	# define q as the exit button
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

