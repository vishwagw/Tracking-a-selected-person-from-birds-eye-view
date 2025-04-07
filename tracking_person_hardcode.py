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

def main():
    # Hardcoded configurations
    video_path = "./input1.mp4"  # CHANGE THIS to your video file path
    output_path = "output.mp4"
    tracker_type = "CSRT"  # Options: CSRT, KCF, MOSSE, MedianFlow
    
    # Initialize video capture with hardcoded path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
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
    tracker = create_tracker(tracker_type)
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
    
    print(f"Tracking complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
