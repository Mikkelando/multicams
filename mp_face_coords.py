import cv2
import mediapipe as mp
import numpy as np
import time


def calc_angles_with_video():

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh =mp_face_mesh.FaceMesh(
        min_detection_confidence=.7,
        min_tracking_confidence=.7
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
                # print("ROT VEC:" , rot_vec)
                # print()
                rmat, jac = cv2.Rodrigues(rot_vec)

                # print(rmat)
                # print()
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360


                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec,
                                                                cam_matrix, dist_matrix)
                
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                

                cv2.line(image, p1, p2, (255, 0, 0 ), thickness=2)

                cv2.putText(image, f'x : {np.round(x, 3)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.putText(image, f'y : {np.round(y, 3)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
                cv2.putText(image, f'z : {np.round(z, 3)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

                print('\n'+ "="*40)
                print('x: ', x)
                print('y: ', y)
                print('z: ', z)
                print('\n'+ "="*40)


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





def calc_angles(file_path, model='mp_face_mesh'):

    if model == 'mp_face_mesh':
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh =mp_face_mesh.FaceMesh(
            min_detection_confidence=.5,
            min_tracking_confidence=.5
            )
    else:
        face_mesh = model

    cap = cv2.VideoCapture(file_path)
    frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    coords = [None for _ in range(total_frames)]
   


    while cap.isOpened:
        ret, image = cap.read()
        if not ret:
            break

        frames += 1
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        coords[frames-1] = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx ==61 or idx == 291 or idx == 199:
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
                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                coords[frames-1] = [x, y, z]
                

    cap.release()


    return (total_frames, coords)




def calc_pose(image_path, model = 'face_mesh'):

    if model == "face_mesh":
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh =mp_face_mesh.FaceMesh(
            min_detection_confidence=.7,
            min_tracking_confidence=.7
            )
    else:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = model

    mp_drawing = mp.solutions.drawing_utils

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    image = cv2.imread(image_path)


    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
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
            
            rmat, jac = cv2.Rodrigues(rot_vec)

        
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360


           
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            

            cv2.line(image, p1, p2, (255, 0, 0 ), thickness=2)

            cv2.putText(image, f'x : {np.round(x, 3)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, f'y : {np.round(y, 3)}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(image, f'z : {np.round(z, 3)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        

        

        # print('FPS', fps)
        

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list = face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec
        )

        pose = [x, y, z]

    else:
        pose = [0, 0, 0]

    return image, pose
    


if __name__ == "__main__":
    # calc_angles_with_video()
    f = calc_angles('video.mp4')
    print(f'f[0] = {f[0]}')
    print(f'f[1] = {f[1][:20]}')