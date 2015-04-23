from __future__ import with_statement
import numpy as np
import cv2
import progressbar as pb
import cPickle
import matplotlib.pyplot as plt
from cv2Adapter import CaptureElement, WriteElement
from sklearn.cluster import KMeans
import seaborn as sns


class setGTVideo(object):
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.point = []

    def on_click(self,event, x,y,flags,param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
            self.point.append((x, y))
            print (x,y)
            
    def run(self):
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.cv.SetMouseCallback('video', self.on_click)
        with CaptureElement(self.video_path) as ce:
            for img in ce.frames:
                cv2.imshow('video', img)
                cv2.waitKey(-1)
    


# Cross correlation tracking
def track(video_path, selector_data):

    def corrMatch(img, selector_data):
        # Matches obj in img
        # Works for color or gray images
        # Returns position of obj in img and confidence measure
        result = cv2.matchTemplate(img,templ = selector_data['obj'],
                                   method = cv2.cv.CV_TM_CCORR_NORMED)
        _,maxVal,_,maxLoc = cv2.minMaxLoc(result)
        return(maxVal, maxLoc)

    def camShiftMatch(img, selector_data):
         
        # image (RGB to  HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 0., 0.)),
                           np.array((180., 255., 32.)))
        # object (RGB to HSV)
        hsv_roi = cv2.cvtColor(selector_data['obj'], 
                               cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(selector_data['obj'], np.array((0., 0., 0.)), 
                               np.array((180., 255., 32.)))
        # histogram of object
        hist = cv2.calcHist( [hsv_roi], [0], None, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist = hist.reshape(-1)
        
        # back projections of object histogram to image
        prob = cv2.calcBackProject([hsv], [0],hist,[0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        _, loc = cv2.CamShift(prob, selector_data['selection'], term_crit)
        
        return(1, loc[:2])

    # Parse data from object selector
    obj = selector_data['obj']
    framePosition = selector_data['framePosition']

    # Track selected object
    location = []
    confidence = []
    Npix = 200
    h,w = obj.shape[:2]
    with CaptureElement(video_path) as ce:
        y1, y2, x1, x2 = (0,ce.frames[framePosition].shape[1],
                          0,ce.frames[framePosition].shape[0])
        pbar = startBar('Tracking',
                        len(ce.frames) - framePosition)
        # start tracking from frame of selected object
        for n,img in enumerate(ce.frames):
            (conf, loc) = corrMatch(img[y1 : y2, x1 : x2, :], selector_data)
            #(conf, loc) = camShiftMatch(img, selector_data)
            selector_data['obj'] = img[y1 : y2, x1 : x2, :][loc[1] : loc[1] + w, loc[0] : loc[0] + h, :]
            loc = (x1+loc[0], y1+loc[1])
            (x,y) = loc
            x1 = max(x - Npix, 0)
            x2 = min(x + w + Npix, img.shape[1])
            y1 = max(y - Npix, 0)
            y2 = min(y + h + Npix, img.shape[0])
            location.append(loc)
            confidence.append(conf)
            pbar.update(n)
        pbar.finish()

    tracker_data = {'location': location,
                    'confidence': confidence}

    return(tracker_data)

def plotResults(video_path, tracker_data):
    
    sns.set(style="white")

    location = tracker_data['location']

    with CaptureElement(video_path) as ce:
        vidH, vidW = ce.frames[0].shape[:2]
    
    # Number of people
    Nsub =2
    colors = ["b","r","g","k"]
    # Cluster positions
    clf = KMeans(n_clusters= Nsub)
    labels = clf.fit_predict(location)
    
    # Plotting results
    ax = plt.subplot(2,2,1)
       
    for i in range(Nsub):
        x = [loc[0] for n,loc in enumerate(location) if labels[n]==i]
        y = [loc[1] for n,loc in enumerate(location) if labels[n]==i]
        ax.plot(x, y, colors[i]+'.-', alpha = 0.5, markersize = 10)
        
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_ylim(vidH,0)
    ax.set_xlim(0,vidW)
    ax.legend(['Student '+str(x) for x in range(Nsub)])
    ax.set_title('Keyboard position', {'fontsize' : 20})
    
    ax = plt.subplot(2,1,2)
    ax.plot(np.arange(len(labels))/30,
            labels,'.-')
    #ax.plot(np.arange(len(labels))/30,
    #        self.correlation,'r.')
    #ax.set_xlabel("Seconds")
    ax.set_yticks([0,1])
    ax.set_ylim(-.5,Nsub -1 + .5)
    ax.set_xlabel('Keyboard possession in time',  {'fontsize' : 15})

    ax = plt.subplot(2,2,2)
    for i in range(Nsub):
        ax.bar(i,np.sum(labels==i)/float(len(labels)), color = colors[i])
    
    ax.set_xticks([0.4, 1.4])
    ax.set_xticklabels(['Student 0', 'Student 1'],  fontdict = {'fontsize' : 15})
    ax.set_title('Keyboard possession',  {'fontsize' : 20})
    plt.show()

class ObjectSelector(object):

    def __init__(self, video_path):
        self.video_path = video_path
        self.drag_start = None
        # Set to rect when the mouse drag finishes
        self.track_window = None
        self.path = video_path

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

    def __enter__(self):
        print( "Keys:\n"
               "    f - forward\n"
               "    b - backward\n"
               "    s - Save cropped image and exit\n"
               "Drag across the object with the mouse\n" )
        cv2.namedWindow('Selected object', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow( "Input Image", cv2.WINDOW_NORMAL )
        cv2.resizeWindow("Input Image", 800, 800)
        cv2.cv.SetMouseCallback( "Input Image", self.on_mouse)
        return self
    
    def __exit__(self, exctype, excinst, exctb):
        cv2.destroyAllWindows()

    def run(self):
        c = 0
        framePosition = 0
        with CaptureElement(self.video_path) as ce:
            while c != ord("s"):
                img  = ce.frames[framePosition]
                tmp = np.copy(img)
                # If mouse is pressed, highlight the current selected
                # rectangle and save selected object
                if self.drag_start and \
                   is_rect_nonzero(self.selection):
                    x,y,w,h = self.selection
                    cv2.rectangle(tmp, pt1 = (x,y),
                                  pt2 = (x+w,y+h),
                                  color = (255,255,255),
                                  thickness = img.shape[1]/300)
                elif self.track_window and \
                     is_rect_nonzero(self.track_window):
                    obj = np.copy(img[y:(y+h),x:(x+w)])
                    cv2.imshow('Selected object', obj)

                cv2.imshow("Input Image", tmp)
                c = cv2.waitKey(7)
                # Go Back one frame
                if c == ord("b"):
                    framePosition -= 1
                    if framePosition<0: 
                        framePosition =0
                # Go fordward one frame    
                if c == ord("f"):
                    framePosition += 1
                    if framePosition>= len(ce.frames):
                        framePosition = len(ce.frames)-1
        selector_data = {'obj' : obj,
                        'framePosition' : framePosition,
                        'selection' : self.selection}

        return (selector_data)   

def distance(x,y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
    
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)
        
def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)

def startBar(title = "Processing", MaxCount = 1e6):
    widgets = [title+":", pb.Percentage(), ' ',
               pb.Bar(marker=pb.RotatingMarker()),' ',
               pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets,
                          maxval=MaxCount).start()
    return pbar

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, color)

def saveDict(filename, d):
    with open(filename,'w') as f:
        f.write(cPickle.dumps(d))

def loadDict(filename):
    return cPickle.load(open(filename))

def saveObject(obj):
    index = []
    for point in obj["Kp"]:
        temp = (point.pt, point.size,
                point.angle, point.response,
                point.octave, point.class_id)
        index.append(temp)

    obj["Kp"]=index
    
    # Dump the Object
    f = open("Object.p", "w")
    f.write(cPickle.dumps(obj))
    f.close()

def loadObject(path):

    obj = cPickle.load(open(path))
    index = obj["Kp"]
    kp = []
 
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],
                            y=point[0][1],
                            _size=point[1],
                            _angle=point[2],
                            _response=point[3],
                            _octave=point[4],
                            _class_id=point[5]) 
        kp.append(temp)

    obj["Kp"]=kp

    return(obj)



def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
    # Convert all images into gray scale
    if len(img1.shape)==3:
        img1 = cv2.cvtColor(img1,cv2.cv.CV_BGR2GRAY)
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2,cv2.cv.CV_BGR2GRAY)
    
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)),
                 (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    plt.imshow(out); plt.show()
    

def writeMatch(video_path, tracker_data):
    
    # Parse tracker data
    location = tracker_data['location']
    confidence = tracker_data['confidence']
    # Set video format
    cap_out = cv2.VideoWriter(video_path + ".out.avi" ,
                              fourcc = cv2.cv.CV_FOURCC(*'XVID'),
                              fps = 30,
                              frameSize = (640,480))

    with CaptureElement(video_path) as ce:
        pbar = startBar("Writing video", len(ce.frames))
        for n,img in enumerate(ce.frames):            
            cv2.circle(img, location[n], radius = 30,
                       color = (0, 0, 255),thickness = -1)
            img = cv2.resize(img, (640, 480))
            cap_out.write(img)
            pbar.update(n)
        pbar.finish()

    cap_out.release()


class OpticalFlow(object):

    def __init__(self, video_path):
        self.video_path = video_path

    def demo(self):
        'Demo for Optical flow, draw flow vectors and write to video'
        
        with CaptureElement(self.video_path) as ce, WriteElement(self.video_path) as we:
            pbar = startBar("Computing flow vectors", len(ce.frames))
            hsv = np.zeros_like(ce.frames[0])
            hsv[...,1] = 255
            for n, img in enumerate(ce.frames):
                # First run
                if n < 1:
                    prev_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                    # Reformatting for display
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    hsv[...,0] = ang*180/np.pi/2
                    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    we.write_frame(rgb)
                    
                    prev_frame = current_frame.copy()
                pbar.update(n)
        pbar.finish()
                
class HistBackProj(object):
    def __init__(self, video_path):
        self.video_path = video_path

    def demo(self):
        'Demo of histogram back projection for Camshift algorithm'
        with ObjectSelector(self.video_path) as osv:
            selector_data = osv.run()
        roi = selector_data['obj']        

        # Set up
        # Disc convolution kernel
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        # histogram of hsv channels from selected object
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        #roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        roihist = cv2.calcHist([hsv],[0, 1], None, [45, 256], [0, 180, 0, 256] )
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)

        with CaptureElement(self.video_path) as ce, WriteElement(self.video_path) as we:
            pbar = startBar("Computing Histogram back projection", len(ce.frames))
            for n, img in enumerate(ce.frames):
                # Current frame hsv 
                hsvt = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
                cv2.filter2D(dst,-1,disc,dst)
                ret,thresh = cv2.threshold(dst,50,255,0)
                thresh = cv2.merge((thresh,thresh,thresh))
                res = cv2.bitwise_and(img,thresh)
                # Write output
                we.write_frame(res)
                pbar.update(n)
        pbar.finish()


class KeyPoint(object):
    def __init__(self, video_path):
        self.video_path = video_path

    def demo(self):
        'Demo of Key Points algorithm'
        surf = cv2.SURF(400)
        surf.hessianThreshold = 5000
        surf.upright = True
        with CaptureElement(self.video_path) as ce, WriteElement(self.video_path) as we:
            pbar = startBar("Computing Key points", len(ce.frames))
            for n, img in enumerate(ce.frames):
                kps, des = surf.detectAndCompute(img, None)
                points = [ map(int, kp.pt) for kp in kps]
                [ cv2.circle(img, tuple(point),
                             radius=10, color=(0,0,255),
                             thickness=-1) for point in points]
                we.write_frame(img)
                pbar.update(n)
        pbar.finish()
