'''
Perspective Correction
Author: Rowel Atienza rowel@eee.upd.edu.ph
Usage: python3 perspective.py
Solves for the Homogenous Matrix H in xp = H*x
xp is the rectified image pixel coordinate
x is the original pixel coordinate
Click 4 pts in the order : top left, top right, bottom left, bottom right
'''

import numpy as np
import numpy.linalg as la
import math

import PIL
from PIL import ImageTk
from tkinter import *
import matplotlib.image as image

from os.path import isfile, join
import argparse

class Settings(object):
    def __init__(self):
        self.image = "batanes.jpg"

class UIFrame(object):
    def __init__(self, parent, settings=Settings()):
        self.settings = settings
        self.parent = parent
        self.ptcount = 0

        self.parent.wm_title("Perspective Correction")
        self.canvas = None
        self.create_canvas()
        self.image = None
        self.imgtks = np.array([])
        self.pts = np.array([])
        self.encodings = None
        self.load_image()


    def display_image(self, data=None, encoding = 'RGB', xorg=0, yorg=0):
        if encoding=='L':
            data = np.reshape(data, [data.shape[0],data.shape[1]])
        img = PIL.Image.fromarray(data, encoding)
        size = data.shape[1], data.shape[0]
        print(size)
        img.thumbnail(size)
        imgtk = ImageTk.PhotoImage(img)
        self.canvas.create_image(xorg, yorg, image=imgtk, anchor="nw")
        self.imgtks = np.append(self.imgtks, imgtk)
        self.canvas.pack(side=LEFT)
        
    def create_canvas(self):
        if self.canvas is None:
            self.canvas = Canvas(self.parent)
        self.canvas.bind("<Button-1>", self.printcoords)
        self.canvas.pack(fill=BOTH, expand=True)

    def load_image(self, xoffset=0, imgarr=None):
        # for image in self.images:
        if imgarr is None:
            print("image file: ", self.settings.image)
            img = np.array(image.imread(self.settings.image))
            self.image = img
            print("image shape: ", self.image.shape)
        else:
            img = imgarr
        if xoffset==0:
            self.size = img.shape[1], img.shape[0]
            self.dsize = 2*img.shape[1], img.shape[0]
        img = PIL.Image.fromarray(img, 'RGB')
        img.thumbnail(self.size)
        imgtk = ImageTk.PhotoImage(img)
        self.canvas.create_image(xoffset,0, image=imgtk, anchor="nw")
        self.imgtks = np.append(self.imgtks, imgtk)
        self.canvas.pack(side=LEFT)
        # x = x + self.xdim
        if xoffset==0:
            dim = "%dx%d+0+0" % (self.dsize)
            self.parent.geometry(dim)

    def draw_rect(self,x,y,dw=4, color="#f00"):
        tlx = x-dw
        tly = y-dw
        brx = x+dw
        bry = y+dw
        self.canvas.create_rectangle(tlx,tly, brx, bry, outline=color)
        self.canvas.pack()

    def rectify(self):
        print(self.pts)
        print(self.ptps)

        A = np.zeros([8,8])
        b = np.zeros([8,1])
        j = 0
        for i in range(0,A.shape[0],2):
            x = self.pts[j][0]
            y = self.pts[j][1]
            xp = self.ptps[j][0]
            yp = self.ptps[j][1]
            j += 1

            A[i][0] = -x
            A[i][1] = -y
            A[i][2] = -1
            A[i][3] = 0
            A[i][4] = 0
            A[i][5] = 0
            A[i][6] = xp*x
            A[i][7] = xp*y
            b[i] = -xp

            A[i+1][0] = 0
            A[i+1][1] = 0
            A[i+1][2] = 0
            A[i+1][3] = -x
            A[i+1][4] = -y
            A[i+1][5] = -1
            A[i+1][6] = x*yp
            A[i+1][7] = y*yp
            b[i+1] = -yp
        print(A)
        print(b)
        h = np.matmul(la.pinv(A),b)
        print(h)
        H = np.ones([3,3])
        H[0][0] = h[0]
        H[0][1] = h[1]
        H[0][2] = h[2]
        H[1][0] = h[3]
        H[1][1] = h[4]
        H[1][2] = h[5]
        H[2][0] = h[6]
        H[2][1] = h[7]
        H[2][2] = 1.0

        w = self.image.shape[1]
        h = self.image.shape[0]
        rimg = np.zeros(self.image.shape) 
        rimg = rimg.astype(np.uint8)
        print("Rectified image.shape, ", rimg.shape)
        for j in range(h):
            for i in range(w):
                x = np.ones([3,1])
                x[0] = i
                x[1] = j
                xp = np.matmul(H,x)
                xp = xp/xp[2][0]
                x = int(round(xp[0][0]))
                y = int(round(xp[1][0]))
                if x>=0 and x<w and y>=0 and y<h:
                    rimg[y][x] = self.image[j][i]

        self.load_image(xoffset=w, imgarr=rimg)

        for i in range(4):
            x = np.ones([3,1])
            x[0] = self.pts[i][0]
            x[1] = self.pts[i][1]
            print("x: ", x)
            xp = np.matmul(H,x)
            xp = xp/xp[2][0]
            x = int(xp[0][0])
            y = int(xp[1][0])
            print("x: ", x)
            print("y: ", y)
            # print("xp: ", xp)

    def printcoords(self, event):
        # outputting x and y coords to console
        print("Event x,y:", event.x, event.y)
        pt = np.array([event.x, event.y])
        pt = np.reshape(pt, [1,-1])
        l = len(self.pts)
        if l==0:
            self.pts = np.zeros([4,2])
            self.ptps = np.zeros([4,2])
            self.ptps[0][0] = 0
            self.ptps[0][1] = 0
            self.ptps[1][0] = self.size[0]
            self.ptps[1][1] = 0
            self.ptps[2][0] = 0
            self.ptps[2][1] = self.size[1]
            self.ptps[3][0] = self.size[0]
            self.ptps[3][1] = self.size[1]
        if self.ptcount<4:
            self.pts[self.ptcount][0] = event.x
            self.pts[self.ptcount][1] = event.y
            self.draw_rect(event.x,event.y)
            self.ptcount += 1
        if self.ptcount==4:
            self.rectify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",\
                        help="Image.")
    
    args = parser.parse_args()
    settings = Settings()
    if args.image:
        settings.image = args.image
    root = Tk()
    frame = UIFrame(root, settings=settings)
    root.mainloop()
