import argparse
import datetime
import imutils
import time
import cv2

#construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="INSERT PATH TO ORIGINAL VIDEO")
ap.add_argument("-a", "--min-area", type=int, default = 2500, help = "minimum area size")
    #minimum size in pixels for a region of an image to be considered motion
args = vars(ap.parse_args())

#read the video file 
old = cv2.VideoCapture("INSERT PATH TO ORIGINAL VIDEO")

#intialize output file
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('INSERT PATH TO OUTPUT VIDEO', fourcc, 30.0, (1024,576))

#initialize first frame
firstFrame = None


while True:
    #grab the current frame and initialize the movement/no movement text(whether there is movement or not)
    (grabbed, frame) = old.read() #.read() returns tuple of whether frame was succesfully 'grabbed' (read) and frame itself
    text = "still" #whether motion is detected

    #if frame could not be grabbed, we reached end of video
    if not grabbed:
        break

    #resize frame, convert to grayscale, blur
    frame = imutils.resize(frame, width=1024, height=576) #resize processed image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale since color doesn't affect our algorithm
    gray = cv2.GaussianBlur(gray, (21, 21), 0) #Gaussian blur to smoothen images (smooth out noise)

    #if first frame is None, intialize it (assume first frame has NO MOTION and is good example of what background looks like)
    if firstFrame is None:
        firstFrame = gray
        continue

    #compute absolute difference between current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray) #absolute difference in pixel intensity between firstFrame and currentFrame
    #threshold of x to reveal regions of image that have significant changes in pixel intensity,
    #>x white, foreground. x> black, background
    x = 25
    thresh = cv2.threshold(frameDelta, x, 255, cv2.THRESH_BINARY)[1] 
    
    #dilate threshold image to fil in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #contour detection

    #loop over contours
    for c in cnts:
        #if contour(of movement) is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        #compute bounding box for contour, draw it on the frame, update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = "moving"
        #record output (TO EDIT)
        out.write(frame)

    #draw text/timestamp
    cv2.putText(frame, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow("frame", frame)
    #threshold/delta for debugging
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)


    if cv2.waitKey(1) * 0xFF == ord('q'):
        break
"""                      
    while(old.isOpened()):
    #reads individual frames, image stored as 'frame' variable, 'ret' is false if 
    ret, frame = old.read() 
    if ret == True:
        
        #flip the frame
        #frame = cv2.flip(frame, flipCode=-1)

        #write the frame
        out.write(frame)
        
        #displays frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
"""

old.release()
out.release()
cv2.destroyAllWindows()