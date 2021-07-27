# importing the necessary libraries
import cv2
import numpy as np
import emotionClassifier


# Enter file path to analyse
filePath = 'emotions.mp4'
cap = cv2.VideoCapture(filePath)

# Analyse emotions from web cam
# cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotionClassifier = emotionClassifier.emotionClassifier()


# Loop until the end of the video
while (cap.isOpened()):

	# Capture frame-by-frame
	ret, frame = cap.read()


	# Display the resulting frame
	cv2.imshow('Frame', frame)

	# conversion of BGR to grayscale is necessary to apply this operation
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	if len(faces) == 0:
		print('No face in frame.')

	for (x,y,w,h) in faces:

		face = frame[y:y+h, x:x+w]
		face = cv2.resize(face, (48, 48), interpolation = cv2.INTER_NEAREST)
		emotionClassifier.predict(face[::-1])




	# define q as the exit button
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

