import streamlit as st
import threading
# from runclass import RunClass


def run_in_thread(func):
        def wrapper(*args, **kwargs):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs)
            thread.start()
        return wrapper

class MainClass():

    def __init__(self):
        self.VIDEOS = []
        self.main()

    def main(self):
        st.title("Video Uploader")

        # File uploader
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        
        if uploaded_file is not None:
            self.VIDEOS.append(uploaded_file)
            self.download_video(uploaded_file)
            # st.video(uploaded_file.read()) 
            

        if st.button("Run Process"):
            self.run_process()

        if len(self.VIDEOS) > 0:
            for video in self.VIDEOS:
                pass
                # st.video(video.read()) 

    def run_process(self):
        # Implement your process logic here
        # For demonstration purposes, we print a message
        print("Process executed!")
        print(self.VIDEOS)

    
    # def run_download(self, uploaded_file):
    #     thread = threading.Thread(target=self.download_video, args=(uploaded_file,))
    #     thread.start()


    @run_in_thread
    def download_video(self, uploaded_file):
        with open(f"{uploaded_file.name}", "wb") as f:
            print('form func:', uploaded_file)
            f.write(uploaded_file.getvalue())
            # st.success(f"Video '{uploaded_file.name}' has been successfully saved.")
            # print(data)
            
            print('success')
            
            return
            


if __name__ == "__main__":
    MainClass()