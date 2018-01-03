import cv2
import glob
import random
import numpy

class FaceTracker:
    STD_WIDTH = 150
    STD_HEIGHT = 150

    #possible emotions in dataset are: normal, happy, sad, sleepy, surprised
    #face animation does not support the sleepy emotion
    #disabling emotions can increase the accuracy
    emotions = ['happy', 'normal', 'surprised', 'sad']

    def __init__(self, datadir):
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        #Open CV has two different facial recognition options.
        #In high amounts of light, I found LBPHF to be more accurate
        self.facialRecognizer = cv2.face.LBPHFaceRecognizer_create()
        #self.facialRecognizer = cv2.face.FisherFaceRecognizer_create()
        self.datadir = datadir
        self.training_data = []
        self.training_labels = []

    def get_face(self, baseImage, cords=False):
        gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
        maxArea = 0
        x, y, w, h = 0, 0, 0, 0
        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x, y, w, h = _x, _y, _w, _h
                maxArea = w*h
        if maxArea > 0 :
            face_cropped = gray[y:y+h, x:x+w]
            test_img = cv2.resize(face_cropped, (self.STD_WIDTH, self.STD_HEIGHT))
            if cords:
                return test_img, (x, y, w, h)
            else:
                return test_img
        else:
            if cords:
                return None, ()
            else:
                return None

    def __get_emotion_set(self, emotion):
        files = glob.glob(self.datadir + "/*." + emotion + ".pgm")
        random.shuffle(files)
        return files

    def __make_data_set(self):
        self.training_data = []
        self.training_labels = []
        for emotion in self.emotions:
            for item in self.__get_emotion_set(emotion):
                image = cv2.imread(item)
                faceImage = self.get_face(image)
                if faceImage is not None:
                    self.training_data.append(faceImage)
                    self.training_labels.append(self.emotions.index(emotion))

    def train(self):
        self.__make_data_set()
        self.facialRecognizer.train(self.training_data, numpy.asarray(self.training_labels))

    def get_sentiment(self, face):
        pred, conf = self.facialRecognizer.predict(face)
        return self.emotions[pred]
