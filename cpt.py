#hwllo


import cv2

#our images or videos
#img_file='carpicture.jpg'
video=cv2.VideoCapture('Pedestrains_dashcam.mp4')

#our pre-trained car classifier.
classifier_file='car_detector.xml'
pedestrain_file='pedestrains_detector.xml'

#create car classifier
car_tracker=cv2.CascadeClassifier(classifier_file)
pedestrains_tracker=cv2.CascadeClassifier(pedestrain_file)


#run forever until car stops or some interruption
while True:

    #read the current frame
    successful_read,frame=video.read()

    if successful_read:
        #convert to grayscale
        grayscaled_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break    
    
    #detect cars and pedestrains
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    pedestrains=pedestrains_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars and the pedestrains
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    for (x,y,w,h) in pedestrains:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)    
    
    #display the spotted image
    cv2.imshow('cars and pedestrain detector',frame)    
    
    #dont autoclose
    key=cv2.waitKey(1)

    #stop if Q or q is pressed
    if key==81 or key==113:
        break

#release the VideoCapture object
video.release()











print('code completed')