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

# main function
def main():
    args = parse_arguments()
    
    # Initialize video capture
    if args.video.isdigit():
        cap = cv2.VideoCapture(int(args.video))
    else:
        cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read first frame.")
        return
    
    # Select bounding box for tracking
    print("Select the person to track and press ENTER when done.")
    print("ESC to quit selection and exit.")
    bbox = cv2.selectROI("Select Person", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Person")
    
    # Initialize tracker
    tracker = create_tracker(args.tracker)
    tracker.init(frame, bbox)
    
    # Tracking loop
    tracking_success = True
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate FPS for display
        if elapsed_time > 0:
            fps_display = frame_count / elapsed_time
        else:
            fps_display = 0
        
        if tracking_success:
            # Update tracker
            tracking_success, bbox = tracker.update(frame)
        
        if tracking_success:
            # Draw bounding box
            x, y, w, h = [int(i) for i in bbox]
            
            # Draw rectangle and highlight the person
            # Create a highlight effect
            highlight = frame.copy()
            cv2.rectangle(highlight, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Apply a semi-transparent overlay to make the person stand out
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw the border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display the person ID and tracking info
            text = f"Tracking: Person"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Display tracking failure message
            cv2.putText(frame, "Tracking failure", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow("Person Tracking", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking complete. Output saved to {args.output}")

# main function call
if __name__ == "__main__":
    main()

