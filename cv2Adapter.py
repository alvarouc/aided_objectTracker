# Video reading adpter for opencv

import cv2

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
        self.oldCap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        
    def __iter__(self):
        return self
    
    def next(self):
        retval, img = self.oldCap.read()
        if retval:
            return img
        else: 
            raise StopIteration()

    def __len__(self):
        return self.oldCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        
    # This is slower than next but more flexible as it can be called
    # with list indexing.
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        else:
            self.oldCap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, index)
        return self.oldCap.read()[1]

