"""

python3 videodemo.py -l=4 -r --weights=saved_models/ResNet56v2_4-layer_weights-200.h5

"""

import tinyssd
import numpy as np
import cv2
from tinyssd import TinySSD
import argparse
from resnet import build_resnet
from viz_boxes import show_boxes
import datetime
from skimage.io import imread
import skimage


class  VideoDemo():
    def __init__(self,
                 detector,
                 width=640,
                 height=480):
        self.detector = detector
        self.width = width
        self.height = height
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(1)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB,True)

    def loop(self):
        font = cv2.FONT_HERSHEY_COMPLEX
        pos = (10,30)
        font_scale = 0.8
        font_color = (0, 0, 0)
        line_type = 1

        while True:
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()
            # img = cv2.resize(img, dsize=(320, 240), 
            # interpolation=cv2.INTER_CUBIC)
            filename = "temp.jpg"
            cv2.imwrite(filename, image)
            #img = image.copy()
            #if np.amax(img) > 1.0:
            #    img = img / 255.0
            img = skimage.img_as_float(imread(filename))
            class_names, rects = self.detector.evaluate(image=img)
            elapsed_time = datetime.datetime.now() - start_time
            hz = 1.0 / elapsed_time.total_seconds()
            hz = "%0.2fHz" % hz
            cv2.putText(image,
                        hz,
                        pos,
                        font,
                        font_scale,
                        font_color,
                        line_type)
            for i in range(len(class_names)):
                rect = rects[i]
                x1 = rect[0]
                y1 = rect[1]
                x2 = x1 + rect[2]
                y2 = y1 + rect[3]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                print(x1, y1, x2, y2, class_names[i])
                cv2.putText(image,
                            class_names[i],
                            (x1, y1-15),
                            font,
                            0.5,
                            (255, 0, 0),
                            line_type)

            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Evaluate model"
    parser.add_argument("-e",
                        "--evaluate",
                        default=False,
                        action='store_true', 
                        help=help_)
    help_ = "Use ResNetv2 as base network"
    parser.add_argument("-r",
                        "--resnet",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Image index"
    parser.add_argument("--image_index",
                        default=0,
                        type=int,
                        help=help_)
    help_ = "Number of layers"
    parser.add_argument("-l",
                        "--layers",
                        default=1,
                        type=int,
                        help=help_)
    help_ = "Batch size"
    parser.add_argument("-b",
                        "--batch_size",
                        default=32,
                        type=int,
                        help=help_)


    args = parser.parse_args()

    if args.resnet:
        tinyssd = TinySSD(n_layers=args.layers,
                          build_basenet=build_resnet,
                          batch_size=args.batch_size)
    else:
        tinyssd = TinySSD(n_layers=args.layers,
                          batch_size=args.batch_size)
    if args.weights:
        tinyssd.load_weights(args.weights)
        videodemo = VideoDemo(detector=tinyssd)
        videodemo.loop()
