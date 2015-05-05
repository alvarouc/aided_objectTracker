from cv2Adapter import CaptureElement
from utils import (
    ObjectSelector, 
    writeMatch, 
    track, 
    plotResults,
    )
import numpy as np
import argparse 

def presentation_demo():
    from utils import (saveDict,
                       loadDict,
                       setGTVideo,
                       OpticalFlow,
                       HistBackProj,
                       KeyPoint,
                       BigVideo)
    # Demos for presentation
    # Optical flow demo
    OpticalFlow(video_path).demo()
    # Histogram back projection demo
    HistBackProj(video_path).demo()
    # Key points demo
    KeyPoint(video_path).demo()
    #Video concatenation for presentation
    video_paths =['/home/aulloa/data/aolme/test.mov',
                  '/home/aulloa/data/aolme/results/test.histBack.avi',
                  '/home/aulloa/data/aolme/results/test.opticalFlow.avi',
                  '/home/aulloa/data/aolme/results/test.keyPoints.avi',
    ]
    BigVideo(video_paths).write()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video tracker')
    parser.add_argument('video_path', help='a path to a video file')
    args = parser.parse_args()
    video_path = args.video_path
    
    # Select and crop an object from video
    with ObjectSelector(video_path) as osv:
        selector_data = osv.run()
    #saveDict('selector.data', selector_data)
    #selector_data = loadDict('selector.data')
    tracker_data = track(video_path, selector_data)
    #saveDict('tracker.data', tracker_data)
    #tracker_data = loadDict('tracker.data')

    # Write video from results
    writeMatch(video_path, tracker_data)
    plotResults(video_path, tracker_data)
    




