from moviepy.editor import *
import os

# 设置视频文件所在的文件夹和音频文件的输出文件夹
video_folder = r'C:\Users\ddxxl\Videos\BiliDown\sleep'
audio_folder = r'C:\Users\ddxxl\Videos\BiliDown\sleep\audio'

# 确保输出文件夹存在
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# 遍历文件夹中的所有文件
for file in os.listdir(video_folder):
    if file.endswith(".mp4"):
        file_path = os.path.join(video_folder, file)
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_file = os.path.join(audio_folder, file.replace('.mp4', '.mp3'))
        audio.write_audiofile(audio_file)
