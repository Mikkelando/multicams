import argparse
import time
import cv2
import mediapipe as mp
import numpy as np

from flash import certain_frame_flash, get_args
from mp_face_coords import calc_angles
from video_plus_audio import create_video_without_audio, extract_audio, overlay_audio


# from moviepy.editor import VideoFileClip

face_mesh =mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=.5,
    min_tracking_confidence=.5
    )


def find_flash(list_of_videos, model = "np_hands"):

    print('started calc of flash')

    flashes = {video: certain_frame_flash(video, model=model) for video in list_of_videos } #take a while

    videos = [list_of_videos[i] for i in range(len(list_of_videos)) if flashes[list_of_videos[i]] >= 0 ]

    data = {video: {'flash': flashes[video], 'subdata': []} for video in videos}
    
    for video in videos:
        tmp_data = calc_angles(video, model=face_mesh)
        data[video]['subdata'] = tmp_data
    
    
    name_for_sound =  max(data, key=lambda video: data[video]['subdata'][0] - data[video]['flash']  )

    print('finished calc of flash')

    return name_for_sound, data


def create_new_stream(data, name_for_sound):
    
    

    number_of_frames = data[name_for_sound]['subdata'][0] - data[name_for_sound]['flash'] + 1
    lifeline = [[] for _ in range(number_of_frames)]

    for video in data:
        

        # for i in range(len(data['video']['subdata'][1])):
        # for i in range(number_of_frames):
        #     if i < len(data[video]['subdata'][1]) - 1:
        #         if i < data[video]['flash'] - 1:
        #             pass
        #         else:
        #             lifeline[i - data[video]['flash'] - 1].append( [data[video]['subdata'][1][i], i, video] )

            
        for j in range(len(data[video]['subdata'][1])):
            if j < data[video]['flash'] - 1:
                pass
            else:
                lifeline[j - data[video]['flash'] + 1 ].append([data[video]['subdata'][1][j], j, video])

    print('n of frames')
    print(number_of_frames)
    print('len lifelines')
    print(len(lifeline))
    return lifeline

    
def metric(pose, coord):
    return np.linalg.norm(np.asarray(pose) -np.asarray(coord))
    

def create_timeline(lifeline, pose, data):
   
    timeline = [None for _ in range(len(lifeline))]


    print("START CALC TIMELINE")
    for i in range(len(lifeline)):
        NAME = ''
        m = 10000
        rel_frame = 0
        f =0
        for pack in lifeline[i]:
            if pack[0]:
                f = 1

                if metric(pose, pack[0]) < m:
                    m = metric(pose, pack[0])
                    NAME = pack[2]
                    rel_frame = pack[1]

            
            else:
                pass
                
        if f:

            timeline[i] = [NAME, rel_frame]
            print(m)
        else:
            # if rel_frame + 1 <  data[NAME]['subdata'][0]:
            #     timeline[i] = [NAME, rel_frame]
            NAME, rel_frame = timeline[i-1]
            if rel_frame + 1 <  data[NAME]['subdata'][0]:
                timeline[i] = [NAME, rel_frame+1]
            else:
                timeline[i] = timeline[i-1]

            print('face not found, replaced frame !!!')


    return timeline



        










'''
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
'''



def main_process(list_of_videos, pose=[0,0,0]):
    args = get_args()

    
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    
    name_for_sound, data = find_flash(list_of_videos, model=hands)
    lifeline = create_new_stream(data, name_for_sound)
    timeline = create_timeline(lifeline, pose, data)
    BIG = create_video_without_audio(timeline, "main_output_without_audio.mp4")  #TEST.mp4
    extract_audio(name_for_sound, 'main_audio.mp3', start_frame=data[name_for_sound]['flash']-1)
    overlay_audio('main_output_without_audio.mp4', "main_audio.mp3", "main_output.mp4")





if __name__ == "__main__":

    # args = get_args()

    
    # cap_width = args.width
    # cap_height = args.height

    # use_static_image_mode = args.use_static_image_mode
    # min_detection_confidence = args.min_detection_confidence
    # min_tracking_confidence = args.min_tracking_confidence

    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(
    #     static_image_mode=use_static_image_mode,
    #     max_num_hands=2,
    #     min_detection_confidence=min_detection_confidence,
    #     min_tracking_confidence=min_tracking_confidence,
    # )


    vid1 = "vid1.mp4"
    vid2 = "vid2.mp4"
    vid3 = "vid3.mp4" #no flash

    list_of_videos = [vid1, vid2, vid3]

    # name_for_sound, data = find_flash(list_of_videos, model=hands)

    # print('____________name_for_sound_____________')
    # print(name_for_sound)

    # print("___________________________________________")
    # print(data)
    # for vid in data:
    #     print('flash', data[vid]['flash'])
    #     print('total_frames', data[vid]['subdata'][0])

    
    # lifeline = create_new_stream(data, name_for_sound)


    # print("_"*20 + 'LIFE LINE' + "_"*20)
    # print(lifeline[:10])
    # print()
    # print()
    # print(lifeline[-1:-10:-1])

    # timeline = create_timeline(lifeline, [0,0,0], data)


    # # frame_list = [("video1.mp4", 100), ("video2.mp4", 200), ("video3.mp4", 300)]
    # # audio_file_path = "audio.mp3" 
    # # start_frame = 500 
    # # output_video_path = "output_video.mp4"
    # print("="*20 + 'TIMELINE' + "="*20)
    
    # time.sleep(5)
    # print(timeline)
    main_process(list_of_videos, pose=[0,0,0])