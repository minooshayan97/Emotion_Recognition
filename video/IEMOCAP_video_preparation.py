import pandas as pd
import glob
import os
from moviepy.editor import VideoFileClip

def extract_time_frames_from_evaluation_files(path, output_dir):
    txt_files = glob.glob(os.path.join(path, "*.txt"))
    for file_name in txt_files:
        f = open(file_name, "r")
        lines = f.readlines()
        time_frames = []
        for i,line in enumerate(lines):
            if line.startswith('['):
                row = line.strip().split('\t')
                #print(row)
                turn = row[1]
                time = row[0][1:-1].split('-')
                time_frames.append([turn, time[0], time[1]])
                #print(time_frames)
            else:
                continue
        time_frames_df = pd.DataFrame(time_frames)
        time_frame_filename = output_dir + file_name.split('\\')[-1]
        time_frames_df.to_csv(time_frame_filename, header=None, index=None, sep=',')
        '''print(time_frame_filename)
        print(len(time_frames_df))'''
    return 


def split_avi(txt_file, video_file, output_dir):
    
    with open(txt_file) as f:
        times = f.readlines()
    times = [x.strip() for x in times] 
    for i, time in enumerate(times):
        turnname = time.split(",")[0]
        starttime = float(time.split(",")[1])
        endtime = float(time.split(",")[2])
        
        target_file = output_dir + turnname + '.mp4'
        extract_subclip(video_file, starttime, endtime, targetname=target_file)    



def extract_subclip(filename, t1, t2, targetname):
    videoclip = VideoFileClip(filename)
    clip = videoclip.subclip(t1, t2)
    no_audio = clip.without_audio()
    no_audio.write_videofile(targetname, codec='libx264') 


