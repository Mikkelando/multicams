import time
import dlib
import cv2
import numpy as np



def to_relative(topnose_pose, face_params, parts):
    x, y, w, h = face_params.left(), face_params.top(), face_params.width(), face_params.height()
    for landmark in parts:
        x_abs, y_abs  = landmark.x, landmark.y
        




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


image = cv2.imread("kudri.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    landmarks = predictor(gray, face)
    print(landmarks)
    topnose = landmarks.part(27)

    landmarks = np.array([[landmark.x, landmark.y] for landmark in landmarks.parts()])
    
    # print(landmarks)
    
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    cv2.circle(image, (topnose.x, topnose.y), 5, (0, 255, 0), -1)

    cv2.imshow("Facial Landmarks", image)

    while True:
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    

print('ended')


