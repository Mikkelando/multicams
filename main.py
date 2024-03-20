import cv2
import streamlit as st
import threading
import json
from get_frames_from_videos import main_process
from mp_face_coords import calc_pose
import mediapipe as mp
from threading import Event

mp_face_mesh = mp.solutions.face_mesh
face_mesh =mp_face_mesh.FaceMesh(
    min_detection_confidence=.5,
    min_tracking_confidence=.5
    )

event = Event()

def run_in_thread(func):
        def wrapper(*args, **kwargs):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
        return wrapper

class MainClass():

    
    
    
    def __init__(self):
        self.VIDEOS = []
        self.pose = [0,0,0]
        self.build()


    def build(self):
        st.title("Multi Cams")


        uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
        
            with open("pose.jpg", "wb") as f:
                f.write(uploaded_image.getvalue())

            
            image_cv2 = cv2.imread("pose.jpg")
            print(type(image_cv2))
            
            
            st.image(image_cv2, caption="Исходное изображение", channels="BGR", use_column_width=True)

                
            img, self.pose = calc_pose('pose.jpg', model = face_mesh)
            
            st.image(img, caption="MESH", use_column_width=True)

        
        uploaded_files = st.file_uploader("Upload a video", type=["mp4", "avi"], accept_multiple_files=True)
        
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                self.VIDEOS.append(uploaded_file.name)
                self.download_video(uploaded_file)


            

        if st.button("Run Process"):
            
            self.run_process()

            # while not event.is_set():
               
            #     # event.wait()
                
            #     print('GET VIDEO BACK!')
            #     video_file = open('main_output.mp4', 'rb')
            #     video_bytes = video_file.read()

            #     st.video(video_bytes)

            #     event.clear()
            #     break
                


       

    # @run_in_thread
    def run_process(self):
        print('STARTED!!!')
        print("POSE________", self.pose)
        try:
            main_process(self.VIDEOS, pose=self.pose)
        except:
            print('ERROR')
        # event.set()
            
        print('GET VIDEO BACK!')
        video_file = open('main_output.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)


        return

    
    @run_in_thread
    def find_pose(self, path, model= 'face_mesh'):
        img, pose = calc_pose(path, model= model)
        return (img, pose)

    @run_in_thread
    def download_video(self, uploaded_file):
        
        with open(f"{uploaded_file.name}", "wb") as f:
            print('form func:', uploaded_file)
            f.write(uploaded_file.getvalue())
            print('success')
                
        return
            


if __name__ == "__main__":
    MainClass()