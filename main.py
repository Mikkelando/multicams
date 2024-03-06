import streamlit as st
import threading
import json
# from external_scarvanger import append_to_json


def run_in_thread(func):
        def wrapper(*args, **kwargs):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
        return wrapper

class MainClass():

    def __init__(self):
        self.VIDEOS = []
        self.build()

    def build(self):
        st.title("Video Uploader")

        
        uploaded_files = st.file_uploader("Upload a video", type=["mp4", "avi"], accept_multiple_files=True)
        
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                self.VIDEOS.append(uploaded_file.name)
                self.download_video(uploaded_file)


            

        if st.button("Run Process"):
            
            self.run_process()

    def run_process(self):
        print("Process executed!")
        print(self.VIDEOS)

    
    


    @run_in_thread
    def download_video(self, uploaded_file):
        
        with open(f"{uploaded_file.name}", "wb") as f:
            print('form func:', uploaded_file)
            f.write(uploaded_file.getvalue())
            print('success')
                
        return
            


if __name__ == "__main__":
    MainClass()