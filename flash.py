import copy
import argparse
import itertools

import cv2 as cv
import mediapipe as mp

from model import KeyPointClassifier



def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--file", type=str, default='video.mp4')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def certain_frame_flash(file_path, model = 'mp_hands' ):
    args = get_args()

    
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

  

    
    cap = cv.VideoCapture(file_path)

    frames = 0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) 
    print('TOTAL FRAMES COUNT: ', total_frames)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    if model == "mp_hands":
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    else:
        hands = model
    keypoint_classifier = KeyPointClassifier()

    keypoint_classifier_labels = ['Open','Close','Pointer','OK']

    
 
    flash = {'Left':None, 'Right':None}
    while True:
        
        
        # key = cv.waitKey(10)
        # if key == 27:  # ESC
        #     break
   
       
        ret, image = cap.read()
        if not ret:
            break

        frames += 1

        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
            
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                flash[handedness.classification[0].label] = keypoint_classifier_labels[hand_sign_id]


            # print(flash)
            if flash['Left'] == 'Pointer' and flash['Right'] == 'OK':
                print('FLASH!')
                print(frames)
                break
            flash['Left'] = None
            flash['Right'] = None

                
        else:
            pass


    cap.release()
    print(frames)
    if frames < total_frames:
        return frames
    else:
        return -10
    # return frames if frames < total_frames else -1



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)


    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



if __name__ == '__main__':
    frame = certain_frame_flash('video.mp4')
    print(frame)