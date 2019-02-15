import argparse
import sys
import time
import cv2

def main():
    video_capture = cv2.VideoCapture('video/output.mp4')
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        cv2.waitKey(1)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
