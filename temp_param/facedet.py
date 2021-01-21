#!/usr/bin/env python



import sys
import rospy
#from matplotlib import pyplot as plt
#import dlib
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge=CvBridge()

def image_callback():
    #print("Got an image")
    #global bridge

    '''try: 
        cv_image=bridge.imgmsg_to_cv2(ros_image,"bgr8")
    except CvBridgeError as e:
        print(e)'''
    cap=cv2.VideoCapture(0)
    

    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        #img = cv2.imread('images.jpeg')
        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #cpyimg=frame
        # Draw rectangle around the faces
        x1=0
        y1=0
        w1=0
        h1=0

        for (x, y, w, h) in faces:                                                      #This for loop is for detecting multiple faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)                    
            #print(x)
            crop_img = frame[y:y+h, x:x+w]
            #cv2.imshow("cropped", crop_img)
            x1=x
            y1=y
            w1=w
            h1=h
        # Display the output
        #crop_img = frame[y1:y1+h1, x1:x1+w1]
        #cv2.imshow("cropped", crop_img)
        # Initiate STAR detector
        orb = cv2.ORB()

        # find the keypoints with ORB
        kp = orb.detect(crop_img,None)

        # compute the descriptors with ORB
        kp, des = orb.compute(crop_img, kp)

        # draw only keypoints location,not size and orientation
        img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
        plt.imshow(img2),plt.show()

        #img=
        #cv2.imshow('img', frame)

    # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
        #cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Load the cascade
    '''face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the input image
    #img = cv2.imread('images.jpeg')
    # Convert into grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    cv2.imshow('img', cv_image)
    cv2.waitKey(3)
    '''
    cap.release()
    cv2.destroyAllWindows()




    
    






    #cv2.imshow("image",cv_image)
    #cv2.waitKey(3)


def main(args):
    #rospy.init_node("image_converter",anonymous=True)
    #for turtlebot3 waffle
    #image_topic="/camera/rgb/image_raw/compressed"
    #for usb cam
    #image_topic="/usb_cam/image_raw"

    #sub=rospy.Subscriber("/zed/left/image_color",Image,image_callback)
    #sub=rospy.Subscriber("/usb_cam/image_raw/",Image,image_callback)
    #sub=rospy.Subscriber("/raspicam_node/image_raw",Image,image_callback)
    #/raspicam_node/image_raw

    image_callback()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
if __name__=='__main__':
    main(sys.argv)
