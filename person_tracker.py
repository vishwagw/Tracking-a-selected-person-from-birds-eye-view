# using argpase library for argument parse:
# importing
import cv2
import numpy as np
import argparse
import time

# passing a command line argument to the script
def parse_arguments():
    parser = argparse.ArgumentParser(description='Track a person in a video.')
    parser.add_argument('--video', type=str, help='Path to input video file.', default=0)
    parser.add_argument('--output', type=str, help='Path to output video file.', default='output.mp4')
    parser.add_argument('--tracker', type=str, choices=['CSRT', 'KCF', 'MOSSE', 'MedianFlow'], 
                        help='Tracker algorithm to use.', default='CSRT')
    
    return parser.parse_args()

# creating the tracker object 
def create_tracker(tracker_type):
    if tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE()
    elif tracker_type == 'MedianFlow':
        return cv2.legacy.TrackerMedianFlow()
    else:
        return cv2.legacy.TrackerCSRT()  # Default to CSRT

