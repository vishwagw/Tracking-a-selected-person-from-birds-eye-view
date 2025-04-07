# using direct video input:
import cv2
import numpy as np
import time

def create_tracker(tracker_type):
    """Create tracker object based on specified type with version compatibility."""
    # Get OpenCV version
    opencv_version = cv2.__version__.split('.')
    major_ver = int(opencv_version[0])
    minor_ver = int(opencv_version[1])
    
    if major_ver >= 4 and minor_ver >= 5:
        # OpenCV 4.5+ (trackers moved to legacy)
        if tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT()
        elif tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE()
        elif tracker_type == 'MedianFlow':
            return cv2.legacy.TrackerMedianFlow()
        else:
            return cv2.legacy.TrackerCSRT()
    else:
        # Older OpenCV versions
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        elif tracker_type == 'MedianFlow':
            return cv2.TrackerMedianFlow_create()
        else:
            return cv2.TrackerCSRT_create()
