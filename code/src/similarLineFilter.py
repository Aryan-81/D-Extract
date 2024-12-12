import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_similar(line1, line2, dist_thresh=10, height_thresh=20, angle_thresh=2):
    """
    Check if two lines are similar based on distance, height difference, and angle difference.
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Calculate the center points of both lines
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2

    # Calculate the Euclidean distance between the center points
    distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    # Calculate the height difference
    height_diff = abs((y2 - y1) - (y4 - y3))

    # Calculate the angle of each line and their difference
    angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
    angle_diff = abs(angle1 - angle2)

    # Normalize angle difference to be within 0-180 degrees
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # Check if all conditions are within the specified thresholds
    return (distance <= dist_thresh and height_diff <= height_thresh and angle_diff <= angle_thresh)

def filter_similar_lines(lines, dist_thresh=10, height_thresh=20, angle_thresh=2):
    """
    Filter out similar lines based on proximity, height difference, and angle difference.
    """
    filtered_lines = []
    for line in lines:
        if all(not is_similar(line, kept_line, dist_thresh, height_thresh, angle_thresh) for kept_line in filtered_lines):
            filtered_lines.append(line)
    return filtered_lines

# Load the image in grayscale
image = cv2.imread('your_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Detect lines using HoughLinesP
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Filter out similar lines
if lines is not None:
    filtered_lines = filter_similar_lines(lines, dist_thresh=10, height_thresh=20, angle_thresh=2)

    # Create a copy of the original image to draw lines on
    line_image = np.copy(image)

    # Draw each filtered line on the image
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]  # Extract the coordinates from the line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw line in green with thickness 2

    # Display the result using matplotlib
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB for correct color display in matplotlib
    plt.title('Filtered Lines')
    plt.axis('off')
    plt.show()
else:
    print("No lines detected.")
