#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(cascade, image, scaleFactor = 1.05):  
    image_copy = image.copy()                                      # creating copy of original
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)           # Coverting it to grayscale
    gray_image = cv2.equalizeHist(gray_image)                           # Histogram equalization

    # detecting faces using the classifier
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 128), 2)

    return image_copy

image = cv2.imread(r"ref_pic.jpg")

#detecting faces
All_faces = detect_faces(face_cascade, image, scaleFactor = 1.009)

# display image
cv2.namedWindow('Faces', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Faces', 400,200)
cv2.imshow('Faces', All_faces)
cv2.waitKey(0)
cv2.destroyAllWindows()

