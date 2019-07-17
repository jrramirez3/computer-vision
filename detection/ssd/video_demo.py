"""

python3 video_demo.py -l=4 -r --weights=saved_models/ResNet56v2_4-layer_weights-200.h5

"""

import ssd
import numpy as np
import cv2
import argparse
import datetime
import skimage
import label_utils
import config

from ssd import SSD
from resnet import build_resnet
from viz_boxes import show_boxes
from skimage.io import imread


class  VideoDemo():
    def __init__(self,
                 detector,
                 camera=0,
                 width=640,
                 height=480):
        self.camera = camera
        self.detector = detector
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

        self.videowriter = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (self.width, self.height), isColor=True)
        # self.capture.set(cv2.CAP_PROP_CONVERT_RGB,True)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        pos = (10,30)
        font_scale = 0.9
        font_color = (0, 0, 0)
        line_type = 1

        while True:
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()

            #filename = "temp.jpg"
            #cv2.imwrite(filename, image)
            #img = skimage.img_as_float(imread(filename))

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            #if np.amax(img) > 1.0:
            #    img = img / 255.0

            class_names, rects = self.detector.evaluate(image=img)
            
            #elapsed_time = datetime.datetime.now() - start_time
            #hz = 1.0 / elapsed_time.total_seconds()
            #hz = "%0.2fHz" % hz
            #cv2.putText(image,
            #            hz,
            #            pos,
            #            font,
            #            font_scale,
            #            font_color,
            #            line_type)

            items = {}
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
                name = class_names[i].split(":")[0]
                if name in items.keys():
                    items[name] += 1
                else:
                    items[name] = 1
                index = label_utils.class2index(name)
                color = label_utils.get_box_rgbcolor(index)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                # print(x1, y1, x2, y2, class_names[i])
                cv2.putText(image,
                            name,
                            (x1, y1-15),
                            font,
                            0.5,
                            color,
                            line_type)

            count = len(items.keys())
            if count > 0:
                xmin = 10
                ymin = 10
                xmax = 220
                ymax = 40 + count * 30
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=-1)

                prices = config.params['prices']
                total = 0.0
                for key in items.keys():
                    count = items[key]
                    cost = count * prices[label_utils.class2index(key)]
                    total += cost
                    display = "%0.2f :%dx %s" % (cost, count, key)
                    cv2.putText(image,
                                display,
                                (xmin + 10, ymin + 25),
                                font,
                                0.55,
                                (0, 0, 0),
                                1)
                    ymin += 30

                cv2.line(image, (xmin + 10, ymin), (xmax - 10, ymin), (0,0,0), 1)
                #cv2.putText(image,
                #            "_________",
                #            (xmin + 10, ymin + 25),
                #            font,
                #            0.55,
                #            (0, 0, 0),
                #            1)
                # ymin += 30

                display = "P%0.2f Total" % (total)
                cv2.putText(image,
                            display,
                            (xmin + 5, ymin + 25),
                            font,
                            0.75,
                            (0, 0, 0),
                            1)

            cv2.imshow('image', image)
            if self.videowriter.isOpened():
                self.videowriter.write(image)
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
    help_ = "Camera index"
    parser.add_argument("--camera",
                        default=0,
                        type=int,
                        help=help_)


    args = parser.parse_args()

    if args.resnet:
        ssd = SSD(n_layers=args.layers,
                  build_basenet=build_resnet,
                  batch_size=args.batch_size)
    else:
        ssd = SSD(n_layers=args.layers,
                  batch_size=args.batch_size)
    if args.weights:
        ssd.load_weights(args.weights)
        videodemo = VideoDemo(detector=ssd, camera=args.camera)
        videodemo.loop()
