import facetrack
import cv2
from tkinter import *
from time import sleep

OUTPUT_SIZE_WIDTH = 640
OUTPUT_SIZE_HEIGHT = 480


class Bot:
    capture = cv2.VideoCapture(0)
    color = "#000000"

    def __init__(self, canvas, facetracker):
        self.canvas = canvas
        self.repeat_emotion_count = 0
        self.emotion = 'normal'
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.facetracker = facetracker

    def update(self):
        rc , fullSizeBaseImage = self.capture.read()
        baseImage = cv2.resize( fullSizeBaseImage, (640, 480))
        testFace, rectCords = self.facetracker.get_face(baseImage, True)
        if testFace is not None:
            self.x , self.y, self.w, self.h = rectCords[0], rectCords[1], rectCords[2], rectCords[3]
            self.emotion = self.facetracker.get_sentiment(testFace)

        print(self.x, self.y, self.w, self.h, self.emotion)

    def __create_eye(self, x, y, r, **kwargs):
        self.canvas.create_oval(x-r, y-r, x+r, y+r, **kwargs)

    def draw(self):
        self.canvas.delete(ALL)
        convX = self.canvas.winfo_width() - self.x - self.w
        thirdX = self.w / 3
        thirdY = self.h / 3

        #draw eyes
        self.__create_eye(convX + thirdX, self.y + thirdY, thirdX/4, fill=self.color)
        self.__create_eye(convX + 2*thirdX, self.y + thirdY, thirdX/4, fill=self.color)

        if self.emotion == 'happy':
            self.canvas.create_arc(convX+thirdX/2, self.y+1.25*thirdY, convX + 5/2*thirdX, self.y+5/2*thirdY, start=190, extent=160, width=thirdY/5, style=ARC, fill=self.color)
        elif self.emotion == 'sad':
            self.canvas.create_arc(convX+thirdX/2, self.y+2*thirdY, convX + 5/2*thirdX, self.y+11/4*thirdY, start=10, extent=160, width=thirdY/5, style=ARC, fill=self.color)
        elif self.emotion == 'surprised':
            self.canvas.create_oval(convX+thirdX/2, self.y+2*thirdY, convX + 5/2*thirdX, self.y+11/4*thirdY, width=thirdY/5, fill=self.color)
        else:
            self.canvas.create_rectangle(convX+thirdX/2, self.y+2*thirdY, convX + 5/2*thirdX, self.y+2.2*thirdY, fill=self.color)

#prep face tracker
facetracker = facetrack.FaceTracker("data")
facetracker.train()

#prep tk window
def quit():
    global master
    master.destroy()

master = Tk()
w = Canvas(master, width=OUTPUT_SIZE_WIDTH, height=OUTPUT_SIZE_HEIGHT)
w.pack()
bot = Bot(w, facetracker)
Button(master, text="Quit", command=quit).pack()

while True:
        sleep(0.08)
        bot.update()
        bot.draw()
        master.update_idletasks()
        master.update()
