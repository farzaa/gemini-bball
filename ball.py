import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Process a basketball video to overlay player stats and information.")
parser.add_argument('--video-path', type=str, required=True, help="Path to the input video file (e.g., final_ball.mp4).")
parser.add_argument('--data-path', type=str, required=True, help="Path to the JSON data file (e.g., ball.json).")
parser.add_argument('--output-path', type=str, required=True, help="Path to the output directory where the processed video will be saved.")
args = parser.parse_args()

# --- Validate and prepare paths ---
if not os.path.isfile(args.video_path):
    print(f"Error: Video file not found at {args.video_path}")
    exit()
if not os.path.isfile(args.data_path):
    print(f"Error: Data file not found at {args.data_path}")
    exit()
if not os.path.isdir(args.output_path):
    print(f"Error: Output directory not found at {args.output_path}. Please create it or provide a valid path.")
    exit()

# Load shot data from JSON
with open(args.data_path, 'r') as f:
    shot_data = json.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video files
# Assuming the same video is used for both processing and display, as per the argument.
# If a separate low-res video is needed for processing, the script logic or arguments might need adjustment.
process_video_path = args.video_path
display_video_path = args.video_path

# Open processing video
process_cap = cv2.VideoCapture(process_video_path)
if not process_cap.isOpened():
    print(f"Error: Could not open video file for processing at {process_video_path}")
    exit()
process_fps = int(process_cap.get(cv2.CAP_PROP_FPS))
process_width = int(process_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
process_height = int(process_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open display video
display_cap = cv2.VideoCapture(display_video_path)
if not display_cap.isOpened():
    print(f"Error: Could not open video file for display at {display_video_path}")
    # Attempt to release the process_cap if it was opened
    if process_cap.isOpened():
        process_cap.release()
    exit()
display_fps = int(display_cap.get(cv2.CAP_PROP_FPS))
display_width = int(display_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
display_height = int(display_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store all processed frames
processed_frames = []

# Animation variables
last_shot_time = None
animation_duration = 1.25  # seconds (changed from 0.5 to 1.0)
current_color = (255, 255, 255)  # Start with white

def parse_timestamp(timestamp):
    # Convert timestamp (e.g., "0:07.5") to seconds
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    # Convert timestamp to frame number
    seconds = parse_timestamp(timestamp)
    return int(seconds * fps)

def wrap_text(text, font, scale, thickness, max_width):
    # Calculate the maximum number of characters that can fit in max_width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_animation_color(elapsed_time, is_made):
    if elapsed_time >= animation_duration:
        return (255, 255, 255)  # Return to white
    
    # Calculate progress through animation (0 to 1)
    progress = elapsed_time / animation_duration
    
    if progress < 0.5:
        # Fade to color
        if is_made:
            # Fade to green (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                255,                            # G
                int(255 * (1 - progress * 2))   # R
            )
        else:
            # Fade to red (BGR format) - keeping some green and blue for brighter red
            return (
                int(255 * (1 - progress * 2)),  # B
                int(255 * (1 - progress * 2)),  # G
                255                             # R
            )
    else:
        # Fade back to white
        if is_made:
            # Fade from green to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                255,                                # G
                int(255 * ((progress - 0.5) * 2))   # R
            )
        else:
            # Fade from red to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                int(255 * ((progress - 0.5) * 2)),  # G
                255                                 # R
            )

# Convert timestamps to frame numbers and add feedback display duration
for shot in shot_data['shots']:
    shot['frame_number'] = timestamp_to_frame(shot['timestamp_of_outcome'], process_fps)
    shot['feedback_end_frame'] = shot['frame_number'] + (4 * process_fps)  # Show feedback for 4 seconds

last_head = None
frame_count = 0
process_every_n_frames = int(process_fps / 20)  # Process at 3 fps
last_shot_result = None

print("Processing video...")

while process_cap.isOpened() and display_cap.isOpened():
    # Read frames from both videos
    process_ret, process_frame = process_cap.read()
    display_ret, display_frame = display_cap.read()
    
    if not process_ret or not display_ret:
        break

    frame_count += 1
    
    # Only process every nth frame
    if frame_count % process_every_n_frames == 0:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get the head landmark (landmark 0 is the top of the head)
            head = results.pose_landmarks.landmark[0]
            # Scale coordinates to display resolution
            head_x = int(head.x * display_width)
            head_y = int(head.y * display_height)
            last_head = (head_x, head_y)

    # Draw the arrow and name if we have a head position
    if last_head is not None:
        head_x, head_y = last_head
        arrow_height = 30  # Increased from 20 to 30
        arrow_width = 45   # Increased from 30 to 45
        arrow_tip_y = max(0, head_y - 110)  # Changed from -60 to -110 to move arrow 50px higher
        # Triangle points for the arrow
        pt1 = (head_x, arrow_tip_y + arrow_height)  # tip
        pt2 = (head_x - arrow_width // 2, arrow_tip_y)  # left
        pt3 = (head_x + arrow_width // 2, arrow_tip_y)  # right
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(display_frame, [pts], (0, 0, 255))  # Red arrow
        # Draw the name above the arrow with black border
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "farza"
        text_size = cv2.getTextSize(text, font, 2.5, 6)[0]  # Reduced from 3.0 to 2.5
        text_x = head_x - text_size[0] // 2
        text_y = arrow_tip_y - 10
        # Black border
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (0, 0, 0), 15, cv2.LINE_AA)  # Reduced from 3.0 to 2.5
        # White fill
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)  # Reduced from 3.0 to 2.5

    # Calculate current shot statistics
    current_shots_made = 0
    current_shots_missed = 0
    current_feedback = None
    
    for shot in shot_data['shots']:
        if shot['frame_number'] <= frame_count:
            if shot['frame_number'] == frame_count:
                # New shot detected
                last_shot_time = time.time()
                last_shot_result = shot['result']
            if shot['result'] == 'made':
                current_shots_made = shot['total_shots_made_so_far']
            else:
                current_shots_missed = shot['total_shots_missed_so_far']
            # Check if we should show feedback for this shot
            if shot['frame_number'] <= frame_count <= shot['feedback_end_frame']:
                current_feedback = shot['feedback']

    # Display shot statistics in top left
    stats_font = cv2.FONT_HERSHEY_SIMPLEX
    stats_border = (0, 0, 0)  # Black border
    stats_scale = 2.1  # Changed from 0.7 to 2.1 (3x larger)
    stats_thickness = 6  # Increased from 2 to 6
    stats_border_thickness = 12  # Increased from 4 to 12
    stats_spacing = 90  # Increased spacing for larger text
    white_color = (255, 255, 255)

    # Position for stats (top left with padding)
    stats_x = 30
    stats_y = 150  # Increased y position to account for larger text

    # Calculate animation color if needed
    if last_shot_time is not None:
        elapsed_time = time.time() - last_shot_time
        if elapsed_time < animation_duration:
            current_color = get_animation_color(elapsed_time, last_shot_result == 'made')
        else:
            current_color = white_color
            last_shot_time = None

    # Draw made shots
    made_text = f"Shots Made: {current_shots_made}"
    
    # Draw made shots with border
    cv2.putText(display_frame, made_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    # Draw made shots with color
    cv2.putText(display_frame, made_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                current_color if last_shot_result == 'made' else white_color, stats_thickness, cv2.LINE_AA)
    
    # Draw missed shots
    missed_text = f"Shots Missed: {current_shots_missed}"
    
    # Draw missed shots with border
    cv2.putText(display_frame, missed_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    # Draw missed shots with color
    cv2.putText(display_frame, missed_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                current_color if last_shot_result == 'missed' else white_color, stats_thickness, cv2.LINE_AA)

    # Display feedback if available
    if current_feedback:
        feedback_font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 1.8  # Changed from 0.6 to 1.8 (3x larger)
        feedback_color = (255, 255, 255)  # White text
        feedback_border = (0, 0, 0)  # Black border
        feedback_thickness = 4  # Increased from 1 to 4
        feedback_border_thickness = 8  # Increased from 3 to 8
        feedback_spacing = 60  # Increased spacing for larger text

        # Wrap text to fit within 80% of screen width
        max_width = int(display_width * 0.8)
        wrapped_lines = wrap_text(current_feedback, feedback_font, feedback_scale, feedback_thickness, max_width)

        # Calculate total height of wrapped text
        total_height = len(wrapped_lines) * feedback_spacing
        start_y = display_height - 90 - total_height  # Increased bottom padding for larger text

        # Draw each line centered
        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            feedback_x = (display_width - text_size[0]) // 2
            feedback_y = start_y + (i * feedback_spacing)

            # Draw text with border
            cv2.putText(display_frame, line, 
                        (feedback_x, feedback_y), feedback_font, feedback_scale, 
                        feedback_border, feedback_border_thickness, cv2.LINE_AA)
            cv2.putText(display_frame, line, 
                        (feedback_x, feedback_y), feedback_font, feedback_scale, 
                        feedback_color, feedback_thickness, cv2.LINE_AA)

    # Store the processed frame
    processed_frames.append(display_frame.copy())

    # Display the frame (optional)
    cv2.imshow('Basketball Player Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
process_cap.release()
display_cap.release()
cv2.destroyAllWindows()

print("Creating final video...")

# Create the final video at normal speed
output_video_name = "processed_video.mp4" # Default name for the output file
# Construct the full output path by joining the output directory and the video name
final_output_path = os.path.join(args.output_path, output_video_name)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
final_out = cv2.VideoWriter(final_output_path, fourcc, display_fps, (display_width, display_height))

# Write all frames to the final video
for frame in processed_frames:
    final_out.write(frame)

final_out.release()

print(f"Processing complete. Final video saved to {final_output_path}")

