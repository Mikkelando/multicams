import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh =mp_face_mesh.FaceMesh(
    min_detection_confidence=.5,
    min_tracking_confidence=.5
    )

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened:
    ret, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx ==61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z*3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])

            face_2d = np.array(face_2d ,dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w   

            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w /2],
                [0,0,1]
            ])
            dist_matrix = np.zeros((4,1), dtype=np.float64)

            ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d,
                                                   cam_matrix, dist_matrix)
            print("ROT VEC:" , rot_vec)
            print()
            rmat, jac = cv2.Rodrigues(rot_vec)

            print(rmat)
            print()
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360


            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec,
                                                             cam_matrix, dist_matrix)
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            # p1 = (int(nose_3d_projection[0]), int(nose_3d_projection[1]))
            # p2 = (int(nose_3d_projection[0] + y * 10), int(nose_3d_projection[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0 ), thickness=2)

            cv2.putText(image, f'x : {np.round(x, 3)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, f'y : {np.round(y, 3)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, f'z : {np.round(z, 3)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            # cv2.putText(image, f'x : {np.round(rot_vec[0]*360, 3)}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # cv2.putText(image, f'y : {np.round(rot_vec[1]*360, 3)}', (199, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # cv2.putText(image, f'z : {np.round(rot_vec[2]*360, 3)}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        # print('FPS', fps)
        cv2.putText(image, f'FPS : {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list = face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec
        )


    cv2.imshow('HEAD', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

