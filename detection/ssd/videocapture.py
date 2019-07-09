"""

python3 videocapture.py --camera=1

"""

import numpy as np
import cv2
import argparse
import datetime


class  VideoCapture():
    def __init__(self,
                 camera=0,
                 width=640,
                 height=480):
        self.camera = camera
        self.width = width
        self.height = height
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def loop(self):

        while True:
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()
            # img = cv2.resize(img, dsize=(320, 240), 
            # interpolation=cv2.INTER_CUBIC)
            image = image / 255.0
            elapsed_time = datetime.datetime.now() - start_time
            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Camera index"
    parser.add_argument("--camera",
                        default=0,
                        type=int,
                        help=help_)


    args = parser.parse_args()

    videocap = VideoCapture(camera=args.camera)
    videocap.loop()
