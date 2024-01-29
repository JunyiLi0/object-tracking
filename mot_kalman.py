# Import libraries
import numpy as np
import cv2
import glob
from scipy.optimize import linear_sum_assignment
from TP1_data.KalmanFilter import KalmanFilter


# Define constants
sigma_iou = 0.5 # IoU threshold for matching
max_age = 1 # Maximum number of frames to keep unmatched tracks

# Define helper functions
def iou(bbox1, bbox2):
    # Compute the intersection-over-union of two bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)

    # Compute the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of the intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Compute the IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou

def assign_detections_to_tracks(detections, tracks):
    # Assign detections to tracks using IoU
    matches = []
    unmatched_detections = []
    unmatched_tracks = []

    # Compute the IoU matrix
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            iou_matrix[d, t] = iou(det, trk)

    # Find the matches using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Filter out the matches with low IoU
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= sigma_iou:
            matches.append((r, c))
        else:
            unmatched_detections.append(r)
            unmatched_tracks.append(c)

    # Add the unmatched detections and tracks
    unmatched_detections += [d for d in range(len(detections)) if d not in row_ind]
    unmatched_tracks += [t for t in range(len(tracks)) if t not in col_ind]

    return matches, unmatched_detections, unmatched_tracks

# Define the Track class
class Track:
    # A Track is a list of consecutive detections of the same object
    def __init__(self, detection, track_id):
        self.detections = [detection] # List of detections
        self.track_id = track_id # Unique ID for the track
        self.age = 0 # Number of frames since the last detection
        self.kalman_filter = KalmanFilter(dt=1, u_x=1, u_y=1, std_acc=0.1, x_std_meas=0.1, y_std_meas=0.1)

    def update(self, detection):
        # Update the track with a new detection
        self.detections.append(detection)
        self.age = 0
        x, y, w, h = detection
        self.kalman_filter.update(np.array([[x + w / 2], [y + h / 2]]))

    def predict(self):
        x, y, _, _ = self.kalman_filter.predict()
        return [x - w / 2, y - h / 2, w, h]

    def increment_age(self):
        # Increment the age of the track
        self.age += 1

# Define the Tracker class
class Tracker:
    # A Tracker is a manager of multiple tracks
    def __init__(self):
        self.tracks = [] # List of tracks
        self.next_id = 1 # Next available ID for a new track

    def update(self, detections):
        # Update the tracker with new detections
        if len(self.tracks) == 0:
            # If no tracks, create new ones for each detection
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
        else:
            # Predict the next detections for each track
            predictions = [trk.predict() for trk in self.tracks]

            # Assign the detections to the tracks
            matches, unmatched_detections, unmatched_tracks = assign_detections_to_tracks(detections, predictions)

            # Update the matched tracks with the assigned detections
            for match in matches:
                self.tracks[match[1]].update(detections[match[0]])

            # Create new tracks for the unmatched detections
            for index in unmatched_detections:
                self.tracks.append(Track(detections[index], self.next_id))
                self.next_id += 1

            # Increment the age of the unmatched tracks
            for index in unmatched_tracks:
                self.tracks[index].increment_age()

            # Remove the tracks that have not been updated for a long time
            self.tracks = [trk for trk in self.tracks if trk.age < max_age]

    def get_tracks(self):
        # Return the current tracks
        return self.tracks

# Create a tracker
tracker = Tracker()

# Load the images
image_paths = sorted(glob.glob("ADL-Rundle-6/img1/*.jpg"))

# Load the detections
detections = np.loadtxt("ADL-Rundle-6/det/det.txt", delimiter=',')

# Loop over the images
for i, image_path in enumerate(image_paths):
    # Read the image
    image = cv2.imread(image_path)

    # Get the detections for the current frame
    frame_detections = detections[detections[:, 0] == i + 1, 2:6]

    # Update the tracker
    tracker.update(frame_detections)

    # Get the tracks
    tracks = tracker.get_tracks()

    # Draw the tracks on the image
    for track in tracks:
        # Get the last detection of the track
        x, y, w, h = track.detections[-1]

        # Draw the bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Draw the track ID
        cv2.putText(image, str(track.track_id), (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Draw the trajectory
        for j in range(len(track.detections) - 1):
            # Get the current and next detection of the track
            x1, y1, _, _ = track.detections[j]
            x2, y2, _, _ = track.detections[j + 1]

            # Draw the line between the detections
            cv2.line(image, (int(x1 + w / 2), int(y1 + h / 2)), (int(x2 + w / 2), int(y2 + h / 2)), (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Image", image)
    cv2.waitKey(1)

# Release the resources
cv2.destroyAllWindows()
