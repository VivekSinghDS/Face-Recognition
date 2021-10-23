import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('train.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('pictures/test.PNG')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)



faceloc = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]
cv2.rectangle(img, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,255), 2)

faceloctest = face_recognition.face_locations(img_test)[0]
encodetest = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255,0,255), 2)


results = face_recognition.compare_faces([encode], encodetest)
faceDistance = face_recognition.face_distance([encode], encodetest)
print(faceDistance)
print(results)
cv2.putText(img_test, f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('name', img)
cv2.imshow('name1', img_test)
cv2.waitKey(0)