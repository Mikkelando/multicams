
import subprocess
import time
import cv2

from moviepy.editor import *
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# from PIL import Image

# ffmpeg_extract_subclip("./library/mySound_Effect.mp4", 6, 8.7, targetname="vid2.mp4")
# intro_sound_vidio_clip = VideoFileClip('test.mp4')


timeline = [
    ['vid2.mp4', 502], ['vid2.mp4', 503], ['vid2.mp4', 504], ['vid2.mp4', 505], ['vid2.mp4', 506], ['vid2.mp4', 507], ['vid2.mp4', 508], ['vid2.mp4', 509], ['vid2.mp4', 510], ['vid2.mp4', 511], ['vid2.mp4', 512], ['vid2.mp4', 513], ['vid2.mp4', 514], ['vid2.mp4', 515], ['vid2.mp4', 516], ['vid2.mp4', 517], ['vid2.mp4', 518], ['vid2.mp4', 519], ['vid2.mp4', 520], ['vid2.mp4', 521], ['vid2.mp4', 522], ['vid2.mp4', 523], ['vid2.mp4', 524], ['vid2.mp4', 525], ['vid2.mp4', 526], ['vid2.mp4', 527], ['vid2.mp4', 528], ['vid2.mp4', 529], ['vid2.mp4', 530], ['vid2.mp4', 531], ['vid2.mp4', 532], ['vid2.mp4', 533], ['vid2.mp4', 534], ['vid2.mp4', 535], ['vid2.mp4', 536], ['vid2.mp4', 537], ['vid2.mp4', 538], ['vid2.mp4', 539], ['vid2.mp4', 540], ['vid2.mp4', 541], ['vid2.mp4', 542], ['vid2.mp4', 543], ['vid2.mp4', 544], ['vid2.mp4', 545], ['vid2.mp4', 546], ['vid2.mp4', 547], ['vid2.mp4', 548], ['vid2.mp4', 549], ['vid2.mp4', 550], ['vid2.mp4', 551], ['vid2.mp4', 552], ['vid2.mp4', 553], ['vid2.mp4', 554], ['vid2.mp4', 555], ['vid2.mp4', 556], ['vid2.mp4', 557], ['vid2.mp4', 558], ['vid2.mp4', 559], ['vid2.mp4', 560], ['vid2.mp4', 561], ['vid2.mp4', 562], ['vid2.mp4', 563], ['vid2.mp4', 564], ['vid2.mp4', 565], ['vid2.mp4', 566], ['vid2.mp4', 567], ['vid2.mp4', 568], ['vid2.mp4', 569], ['vid2.mp4', 570], ['vid2.mp4', 571], ['vid2.mp4', 572], ['vid2.mp4', 573], ['vid2.mp4', 574], ['vid2.mp4', 575], ['vid2.mp4', 576], ['vid2.mp4', 577], ['vid2.mp4', 578], ['vid2.mp4', 579], ['vid2.mp4', 580], ['vid1.mp4', 450], ['vid2.mp4', 582], ['vid2.mp4', 583], ['vid2.mp4', 584], ['vid1.mp4', 454], ['vid1.mp4', 455], ['vid2.mp4', 587], ['vid1.mp4', 457], ['vid1.mp4', 458], ['vid1.mp4', 459], ['vid1.mp4', 460], ['vid1.mp4', 461], ['vid1.mp4', 462], ['vid1.mp4', 463], ['vid1.mp4', 464], ['vid1.mp4', 465], ['vid1.mp4', 466], ['vid1.mp4', 467], ['vid1.mp4', 468], ['vid1.mp4', 469], ['vid1.mp4', 470], ['vid1.mp4', 471], ['vid1.mp4', 472], ['vid1.mp4', 473], ['vid1.mp4', 474], ['vid1.mp4', 475], ['vid1.mp4', 476], ['vid1.mp4', 477], ['vid1.mp4', 478], ['vid1.mp4', 479], ['vid1.mp4', 480], ['vid1.mp4', 481], ['vid1.mp4', 482], ['vid1.mp4', 483], ['vid1.mp4', 484], ['vid1.mp4', 485], ['vid1.mp4', 486], ['vid1.mp4', 487], ['vid1.mp4', 488], ['vid1.mp4', 489], ['vid1.mp4', 490], ['vid1.mp4', 491], ['vid1.mp4', 492], ['vid1.mp4', 493], ['vid1.mp4', 494], ['vid1.mp4', 495], ['vid1.mp4', 496], ['vid1.mp4', 497], ['vid1.mp4', 498], ['vid1.mp4', 499], ['vid1.mp4', 500], ['vid1.mp4', 501], ['vid1.mp4', 502], ['vid1.mp4', 503], ['vid1.mp4', 504], ['vid1.mp4', 505], ['vid1.mp4', 506], ['vid1.mp4', 507], ['vid1.mp4', 508], ['vid1.mp4', 509], ['vid1.mp4', 510], ['vid1.mp4', 511], ['vid1.mp4', 512], ['vid1.mp4', 513], ['vid1.mp4', 514], ['vid1.mp4', 515], ['vid1.mp4', 516], ['vid1.mp4', 517], ['vid1.mp4', 518], ['vid1.mp4', 519], ['vid1.mp4', 520], ['vid1.mp4', 521], ['vid1.mp4', 522], ['vid1.mp4', 523], ['vid1.mp4', 524], ['vid1.mp4', 525], ['vid1.mp4', 526], ['vid1.mp4', 527], ['vid1.mp4', 528], ['vid1.mp4', 529], ['vid1.mp4', 530], ['vid1.mp4', 531], 
['vid1.mp4', 532], ['vid1.mp4', 533], ['vid1.mp4', 534], ['vid1.mp4', 535], ['vid2.mp4', 667], ['vid2.mp4', 668], ['vid2.mp4', 669], ['vid2.mp4', 670], ['vid2.mp4', 671], ['vid2.mp4', 672], ['vid2.mp4', 673], ['vid2.mp4', 674], ['vid2.mp4', 675], ['vid2.mp4', 676], ['vid2.mp4', 677], ['vid2.mp4', 678], ['vid2.mp4', 679], ['vid2.mp4', 680], ['vid2.mp4', 681], ['vid2.mp4', 682], ['vid2.mp4', 683], ['vid2.mp4', 684], ['vid2.mp4', 685], ['vid2.mp4', 686], ['vid2.mp4', 687], ['vid2.mp4', 688], ['vid2.mp4', 689], ['vid2.mp4', 690], ['vid1.mp4', 560], ['vid1.mp4', 561], ['vid1.mp4', 562], ['vid1.mp4', 563], ['vid1.mp4', 564], ['vid1.mp4', 565], ['vid1.mp4', 566], ['vid1.mp4', 567], ['vid1.mp4', 568], ['vid1.mp4', 569], ['vid1.mp4', 570], ['vid1.mp4', 571], ['vid1.mp4', 572], ['vid1.mp4', 573], ['vid1.mp4', 574], ['vid1.mp4', 575], ['vid1.mp4', 576], ['vid1.mp4', 577], ['vid1.mp4', 578], ['vid2.mp4', 710], ['vid2.mp4', 711], ['vid2.mp4', 712], ['vid2.mp4', 713], ['vid2.mp4', 714], ['vid2.mp4', 715], ['vid2.mp4', 716], ['vid2.mp4', 717], ['vid2.mp4', 718], ['vid2.mp4', 719], ['vid2.mp4', 720], ['vid2.mp4', 721], ['vid2.mp4', 722], ['vid2.mp4', 723], ['vid2.mp4', 724], ['vid2.mp4', 725], ['vid2.mp4', 726], ['vid2.mp4', 727], ['vid1.mp4', 597], ['vid2.mp4', 729], ['vid1.mp4', 599], ['vid2.mp4', 731], ['vid1.mp4', 601], ['vid1.mp4', 602], ['vid1.mp4', 603], ['vid1.mp4', 604], ['vid1.mp4', 605], ['vid1.mp4', 606], ['vid1.mp4', 607], ['vid1.mp4', 608], ['vid1.mp4', 609], ['vid1.mp4', 610], ['vid1.mp4', 611], ['vid1.mp4', 612], ['vid1.mp4', 613], ['vid1.mp4', 614], ['vid1.mp4', 615], ['vid1.mp4', 616], ['vid1.mp4', 617], ['vid1.mp4', 618], ['vid1.mp4', 619], ['vid1.mp4', 620], ['vid1.mp4', 621], ['vid1.mp4', 622], ['vid1.mp4', 623], ['vid1.mp4', 624], ['vid1.mp4', 625], ['vid1.mp4', 626], ['vid1.mp4', 627], ['vid1.mp4', 628], ['vid1.mp4', 629], ['vid1.mp4', 630], ['vid1.mp4', 631], ['vid1.mp4', 632], ['vid1.mp4', 633], ['vid1.mp4', 634], ['vid1.mp4', 635], ['vid1.mp4', 636], ['vid1.mp4', 637], ['vid1.mp4', 638], ['vid1.mp4', 639], ['vid1.mp4', 640], ['vid1.mp4', 641], ['vid1.mp4', 642], ['vid1.mp4', 643], ['vid1.mp4', 644], ['vid1.mp4', 645], ['vid1.mp4', 646], ['vid1.mp4', 647], ['vid1.mp4', 648], ['vid1.mp4', 649], ['vid1.mp4', 650], ['vid1.mp4', 651], ['vid1.mp4', 652], ['vid1.mp4', 653], ['vid1.mp4', 654], ['vid1.mp4', 655], ['vid1.mp4', 656], ['vid1.mp4', 657], ['vid1.mp4', 658], ['vid1.mp4', 659], ['vid1.mp4', 660], ['vid2.mp4', 792], ['vid2.mp4', 793], ['vid2.mp4', 794], ['vid2.mp4', 795], ['vid2.mp4', 796], ['vid2.mp4', 797], ['vid2.mp4', 798], ['vid2.mp4', 799], ['vid2.mp4', 800], ['vid2.mp4', 801], ['vid2.mp4', 802], ['vid2.mp4', 803], ['vid2.mp4', 804], ['vid2.mp4', 805], ['vid2.mp4', 806], ['vid2.mp4', 807], ['vid2.mp4', 808], ['vid2.mp4', 809]
]

print(len(timeline))


# frame_list = [("video1.mp4", 100), ("video2.mp4", 200), ("video3.mp4", 300)]
# name_for_sound = "audio.mp3" 
# start_frame = 500 
# output_video_path = "output_video.mp4"



def create_video_with_external_audio(frame_list, output_video_path):
    
    # video_clip = VideoFileClip(audio_file_path)
    # audio_clip = video_clip.audio

    # audio_clip = audio_clip.set_start(t=(start_frame / video_clip.fps))
    BIG = []

    f = 1
    # out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
  

    for video_name, frame_number in frame_list:

        # print(video_name, frame_number)
        
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_name}")
            continue

        if f:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'X264'), fps, (width, height))
            # out2 = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), int(fps), (width, height) )
            # out3 = cv2.VideoWriter(
            #     'ANOTHER_VIDEO.mp4',
            #     FOURCC,
            #     FRAMES_PER_SECOND,
            #     RESOLUTION)
            print( fps, (width, height))
            f = 0
  
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

     
        ret, frame = cap.read()
        # print(frame)


        if not ret:
            print(f"Не удалось прочитать кадр из видео: {video_name}")
            continue
        
        BIG.append(frame)

        
        # if frame.shape[:2] != (480, 640):
        #     frame = cv2.resize(frame, (640, 480))

  
        out.write(frame)
        # out2.write(frame)
        
        cap.release()

    out.release()



    
    return BIG
    

   

# def get_audio(file_name):
#     command = "ffmpeg -i C:/Users/msmkl/PROJECTS_PY/multicams/vid2.mp4 -ab 160k -ac 2 -ar 44100 -vn JUST_AUDIO.mp3"

#     subprocess.call(command, shell=True)



def extract_audio(video_file, output_audio_file, start_frame=503, output_video_path='TEST.mp4'):
    video_clip = VideoFileClip(video_file)
    

    start_time = start_frame / video_clip.fps
    end_time = video_clip.duration

    # audio_clip = video_clip.audio

    print ((start_frame / video_clip.fps))
    # audio_clip = audio_clip.set_start(t = (start_frame / video_clip.fps) ) 

    audio_clip = video_clip.subclip(start_time, end_time).audio

    # video_clip = VideoFileClip(output_video_path)
    # video_clip = video_clip.set_audio(audio_clip)

    # video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    audio_clip.write_audiofile(output_audio_file)

def overlay_audio(video_file, audio_file, output_file):


    # videoclip = VideoFileClip(output_video_path)
    # audioclip = AudioFileClip(output_audio_file)

    # new_audioclip = CompositeAudioClip([audioclip])
    # videoclip.audio = new_audioclip

    # videoclip.write_videofile(output_video_path)
    # Загружаем видео и аудио
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)

    # Выравниваем длину аудио и видео
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)
    elif audio_clip.duration < video_clip.duration:
        video_clip = video_clip.subclip(0, audio_clip.duration)

    # Накладываем аудио на видео
    video_clip_tmp = video_clip.set_audio(audio_clip)
    video_clip_tmp.audio = audio_clip
    final_clip = CompositeVideoClip([video_clip_tmp])

    # Сохраняем итоговый видеофайл
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

# Пример использования функции



    




if __name__ == "__main__": 

    BIG = create_video_with_external_audio(timeline, "TEST.mp4")
    # # create_video_with_external_audio(timeline, 'WIN_20240319_22_57_58_Pro.mp4', 503, "TEST.mp4")
    # # get_audio('vid2.mp4')
    print('done')

    '''
    # print(len(BIG))
    # print(len(BIG[0]))

    # i = 0
    # while True:
    #     cv2.imshow('AAAAAAAAA', BIG[i % len(BIG)])
    #     if cv2.waitKey(27) & 0xFF == ord('s'): 
    #         break
    #     i += 1

    # writer = cv2.VideoWriter('ANOTHER_VIDEO.mp4', cv2.VideoWriter_fourcc(*'X264'), 30, (464, 848))

    # for frame in BIG:
    #     writer.write(frame)
    
    # writer.release()
    '''
    
    
    extract_audio("vid2.mp4", "TEST_AUDIO.mp3")

    overlay_audio("TEST.mp4", "TEST_AUDIO.mp3", "GOOD_FILE.mp4")