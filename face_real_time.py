import cv2
import numpy as np
import face_recognition
import os

path = 'pictures'
images = []
classnames = []

mylist = os.listdir(path)
for cl in mylist:
    curImage = cv2.imread(f'{path}/{cl}')
    images.append(curImage)
    classnames.append(os.path.splitext(cl)[0])
print(len(images))
print(classnames)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encode_list = findEncodings(images)
print(len(encode_list))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceframe = face_recognition.face_locations(imgS)
    encode = face_recognition.face_encodings(imgS, faceframe)

    for encodingface, faceloc in zip(encode, faceframe):
        matches = face_recognition.compare_faces(encode_list, encodingface)
        facedist = face_recognition.face_distance(encode_list, encodingface)
        print(facedist)
        matchIndex = np.argmin(facedist)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)