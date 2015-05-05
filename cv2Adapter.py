# Video reading adpter for opencv

import cv2

class WriteElement(object):
    
    def __init__(self, video_path):
        self.video_path = video_path
        
    def __enter__(self):
        self.cap = cv2.VideoWriter(self.video_path + "_out.avi" ,
                                   fourcc = ('X','V','I','D'),
                                   fps = 30,
                                   frameSize = (640,480))
        return self

    def __exit__(self, exctype, excinst, exctb):
        self.cap.release()

    def write_frame(self, img):
        self.cap.write(cv2.resize(img, (640,480)))
        

class CaptureElement(object):
    
    def __init__(self, video_path):
        self.videoPath = video_path
        self.oldCap = cv2.VideoCapture(video_path)
        
    @property
    def frames(self):
        try:
            return Frame(self.oldCap)
        except :
            pass
    
    def __enter__(self):
        return self
        
    def __exit__(self, exctype, excinst, exctb):
        self.oldCap.release()
        
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.videoPath)

class Frame(object):
    
    def __init__(self,oldCap):
        self.oldCap = oldCap
        self.oldCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    def __iter__(self):
        return self
    
    def next(self):
        retval, img = self.oldCap.read()
        if retval:
            return img
        else: 
            raise StopIteration()

    def __len__(self):
        return self.oldCap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    # This is slower than next but more flexible as it can be called
    # with list indexing.
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        else:
            self.oldCap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.oldCap.read()[1]

