import time
import glob
import shutil
import os

from Helpers.log import Paths

def SaveRecording():
    time.sleep(5)
    try:
        recordingFile = glob.glob(r'*.avi')[0]
        shutil.move(recordingFile, fr'{Paths.execution}\recording.avi')
    except Exception as e:
        print(e)