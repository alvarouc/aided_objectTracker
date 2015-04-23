import cv2
import numpy as np
import sys
import progressbar as pb
import multiprocessing as mp
import functools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils 
# histogram correlation threshold
HIST_TH = 0.8
    
class ObjectSelector:
    # This class allows to manually crop an image from a frame of a
    # video
    # Usage:
    # - Set the video path in the video_dir 
    # - call run() to proceed with manual selection
    # - Select the sub-image with the mouse and then press n
    #   to finish
    # - The cropped image will be saved in self.object
    def __init__(self, video_path):
        # variables that store the big Image and selection
        self.obj =None
        self.objHist = None
        # Set to (x,y) when mouse starts drag
        self.drag_start = None
        # Set to rect when the mouse drag finishes
        self.track_window = None
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        # variables that store the position of the object to track and
        # the confidence level of the object being in such position
        self.position = []
        self.correlation = []
        self.histCor= []
        # Total number of frames
        self.nFrames = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.startFrame = None
        self.Npix = 50

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if event == cv2.cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)
        
    def selectObject(self):

        print( "Keys:\n"
               "    f - forward\n"
               "    b - backward\n"
               "    s - Save cropped image\n"
               "Drag across the object with the mouse\n" )

        cv2.namedWindow( "Input Image", cv2.WINDOW_NORMAL )
        cv2.resizeWindow("Input Image",800,800)
        cv2.moveWindow("Input Image",0,0)
        cv2.namedWindow( "object", cv2.WINDOW_AUTOSIZE )
        cv2.moveWindow("object", 800,0)
        
        # the function on_mouse activates when mouse is over the
        # selectObject window
        cv2.cv.SetMouseCallback( "Input Image", self.on_mouse)
        dum = 0
        _, img = self.cap.read()
        while(img is None):
            _, img= self.cap.read()
            dum  = dum +1
        print "Dropped %d frames"%(dum)    
        
        tempObject = None
        while(True):
            tempImg =  np.copy(img)
            # If mouse is pressed, highlight the current selected
            # rectangle and save selected object
            if self.drag_start and \
               is_rect_nonzero(self.selection):
                x,y,w,h = self.selection
                cv2.rectangle(tempImg, pt1= (x,y),
                              pt2 = (x+w,y+h),
                              color = (255,255,255),
                              thickness = img.shape[1]/300)
            elif self.track_window and \
                 is_rect_nonzero(self.track_window):
                tempObject = np.copy(img[y:(y+h),x:(x+w)])

            # Drawing images
            cv2.imshow("Input Image",tempImg)
            if not tempObject is None :
                cv2.imshow("object",tempObject)
            # Waiting for confirmation of cropped image
            c = cv2.waitKey(7)
            # Go Back one frame
            if c == ord("b"):
                nFrame = self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,nFrame-2)
                _,img = self.cap.read()
            # Go fordward one frame    
            if c == ord("f"):
                _,img = self.cap.read()
                nFrame = self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                # Check for end of video
                if nFrame >= self.nFrames:
                    self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,nFrame-1)
            # Save selected object    
            if c == ord("s"):
                self.obj = tempObject
                self.startFrame = self.cap.get(\
                                  cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                
                # Compute histogram of the image
                self.objHist = cv2.calcHist(self.obj,[0,1,2],
                                            None,[8,8,8],
                                        [0,256,0,256,0,256]).flatten()
                np.save("objectHist.npy",self.objHist)
                np.save("object.npy", self.obj)
                np.save("start.npy", self.startFrame)
                print("Object saved ...")
                break    
        cv2.destroyAllWindows()

    def rewind(self):
        if self.startFrame is None:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)
        else:
            self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,self.startFrame)

    def processFrame(self):
        _, img = self.cap.read()
        # Object dimensions
        h,w = self.obj.shape[:2]
        imgGray = None#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if not self.position:
            result = cv2.matchTemplate(img,templ = self.obj,
                                       method = cv2.cv.CV_TM_CCORR_NORMED)
            _,maxVal,_,maxLoc = cv2.minMaxLoc(result)
            th = 1
        else:

            # Set the search bounds
            x,y = self.position[-1]
            x1 = max(x-self.Npix,0)
            x2 = min(x+w+self.Npix,img.shape[1])
            y1 = max(y-self.Npix,0)
            y2 = min(y+h+self.Npix,img.shape[0])
            # search within bounds
            result = cv2.matchTemplate(img[y1:y2,x1:x2,:],
                                    templ = self.obj,
                                    method = cv2.cv.CV_TM_CCORR_NORMED)
            _,maxVal,_,maxLoc = cv2.minMaxLoc(result)
            # adjust relative coordinates
            maxLoc = (x1+maxLoc[0],y1+maxLoc[1])
            # Get object on the match position
            x,y = maxLoc
            temp = img[y:(y+h),x:(x+w)]
            
            # Compare the histogram of the match with original image
            hTest = cv2.calcHist(temp,[0,1,2],
                                 None,[8,8,8],
                                 [0,256,0,256,0,256]).flatten()
            th = cv2.compareHist(self.objHist,hTest,
                                 cv2.cv.CV_COMP_CORREL)
            # If the original histogram matches the current matched
            # object then use the new object as template
            if th>HIST_TH:
                self.obj = np.copy(temp)
            else:
                maxLoc = self.position[-1]


        self.position.append(maxLoc)
        self.correlation.append(maxVal)        
        self.histCor.append(th)
        return(img, imgGray)

    def track(self):
        # Initialize Progress Bar
        pbar = startBar("Tracking",self.nFrames-self.startFrame)
        # rewind video
        self.rewind()
        
        # Initialize first frame results
        prev, prevGray = self.processFrame()

        currentPos = 0 
        while(currentPos < self.nFrames):
            # Read current frame
            img, gray= self.processFrame()

            # compute optical flow
            #flow = cv2.calcOpticalFlowFarneback(prevGray, gray,
            #                                   0.5, 3, 15, 3,
            #                                   5, 1.2, 0)
            #prevGray = gray

            # chekc if end of video
            currentPos = self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            pbar.update(currentPos-self.startFrame)

        pbar.finish()

    def write(self):
        print "Writing results to video..."
        # - initialize progress bar
        pbar = startBar("Writing Video", self.nFrames)

        cap_out = cv2.VideoWriter(self.path[:-3] + "out.avi" ,
                                  fourcc = cv2.cv.CV_FOURCC(*'XVID'),
                                  fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS),
                                  frameSize = (640,480))
        # rewind input video
        self.rewind()

        counter = 0
        maxCounter = len(self.position)
        currentPos = 0
        while(currentPos<self.nFrames):
            _,img = self.cap.read()
            # Draw a rectangle on matched point
            if (counter < maxCounter):
                x,y = self.position[counter]
                h,w = self.obj.shape[:2]

                # Draw rectangle to detected object
                if self.histCor[counter]>HIST_TH:
                    colorx = (255,255,255)
                else:
                    colorx= (0,0,255)
                cv2.rectangle(img, pt1 = (x,y),
                              pt2 = (x+w,y+h),
                              color = colorx,
                              thickness = img.shape[1]/300)
                # Draw rectangle of movement bounds
                cv2.rectangle(img, pt1 = (x-self.Npix,y-self.Npix),
                              pt2 = (x+w+self.Npix,y+h+self.Npix),
                              color = (255,0,0),
                              thickness = img.shape[1]/300)
            counter = counter +1
            img = cv2.resize(img,(640,480))
            # Write image to video
            cap_out.write(img)
            currentPos = self.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            pbar.update(currentPos-self.startFrame)
           

        pbar.finish()
        cap_out.release()
        print("Done")

    def release(self):
        self.cap.release()        

    def plotResults(self):
        data = np.array(self.position)
        np.save("data.npy",data)
        vidW = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        vidH = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

        # Number of people
        Nsub =2
        colors = ["b","r","g","k"]
        # Cluster positions
        clf = KMeans(n_clusters= Nsub)
        labels = clf.fit_predict(data)
        
        # Plotting results
        ax = plt.subplot(2,2,1)
        for i in range(Nsub):
            ax.plot(data[labels==i,0],
                    data[labels==i,1],
                    colors[i]+'.-')
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_ylim(vidH,0)
        ax.set_xlim(0,vidW)
        ax.legend([str(x) for x in range(Nsub)])
        
        ax = plt.subplot(2,1,2)
        ax.plot(np.arange(len(labels))/self.cap.get(cv2.cv.CV_CAP_PROP_FPS),
                labels,'.')
        ax.plot(np.arange(len(labels))/self.cap.get(cv2.cv.CV_CAP_PROP_FPS),
                self.correlation,'r.')
        ax.set_xlabel("Seconds")
        ax.set_yticks([0,1])
        ax.set_ylim(-.5,Nsub + .5)

        ax = plt.subplot(2,2,2)
        ax.bar(range(Nsub),
               [np.sum(labels==x)/float(len(labels)) for x in range(Nsub)])
        ax.set_xticks(0.4 + np.arange(Nsub))
        ax.set_xticklabels([str(x) for x in range(Nsub)])
        plt.show()

        
if __name__=="__main__":

    # Select an object browsing a video
    keyboard = ObjectSelector(sys.argv[1])
    
    keyboard.selectObject()
    #keyboard.obj = np.load("object.npy")
    #keyboard.startFrame = np.load("start.npy")
    #keyboard.objHist = np.load("objectHist.npy")
    
    keyboard.track() # Track
    #keyboard.removeBadMatch(0.9)
    keyboard.write() # Write results
    ## Object location coordinates
    keyboard.plotResults()
    
    keyboard.release() # clean up
