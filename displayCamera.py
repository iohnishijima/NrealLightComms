import cv2
import numpy as np

displayList = ["2d", "3d"]
flag = "2d"

# Camera intrinsic parameters
camera_matrix = np.array([
    [2.32740900e+03, 0.00000000e+00, 1.31873206e+03],
    [0.00000000e+00, 2.34040733e+03, 9.53409002e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([[0.3134004, -1.8312907, -0.02132312, 0.00495278, 3.20735859]], dtype=np.float32)

def clip_image(image, offset_x, offset_y):
    """
    Function to clip the camera image to 1920x1080.
    
    Parameters:
    - image: The original image
    - offset_x: Horizontal offset (negative left, positive right)
    - offset_y: Vertical offset (negative up, positive down)
    
    Returns:
    - The clipped image
    """
    # Original image size
    original_height, original_width = image.shape[:2]
    
    # Determine the clipping area based on the post-clip size and offsets
    start_x = max(0, min(original_width - 1920, (original_width - 1920) // 2 + offset_x))
    start_y = max(0, min(original_height - 1080, (original_height - 1080) // 2 + offset_y))
    end_x = start_x + 1920
    end_y = start_y + 1080
    
    # Clip the image
    clipped_image = image[start_y:end_y, start_x:end_x]
    
    return clipped_image

def clip_and_combine_image(image, offset_left_x, offset_left_y, offset_right_x, offset_right_y):
    """
    Function to clip the camera image for left and right eyes and combine them into a single frame.
    
    Parameters:
    - image: The original image
    - offset_left_x: Horizontal offset for the left eye image
    - offset_left_y: Vertical offset for the left eye image
    - offset_right_x: Horizontal offset for the right eye image
    - offset_right_y: Vertical offset for the right eye image
    
    Returns:
    - The combined image
    """
    # Clip the image for the left eye
    clipped_left = clip_image(image, offset_left_x, offset_left_y)
    
    # Clip the image for the right eye
    clipped_right = clip_image(image, offset_right_x, offset_right_y)
    
    # Combine both images side by side
    combined_frame = np.hstack((clipped_left, clipped_right))
    
    return combined_frame

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

src_points = np.float32([[0, 0], [2592, 0], [0, 1944], [2592, 1944]]) # Points before transformation (corners of the image)
dst_points = np.float32([[2592 * 0.15, 1944 * 0.15], [2592 * 0.85, 1944 * 0.15], [2592 * 0.15, 1944 * 0.85], [2592 * 0.85, 1944 * 0.85]]) # Points after transformation

try:
    while True:
        # Read image from camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Correct image distortion
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        if flag == "2d":
            # Set clipping offsets (here set to 0, 0 as an example, change as needed)
            offset_x = 22  # Horizontal offset
            offset_y = 150  # Vertical offset
            
            # Clip the image
            clipped_frame = clip_image(undistorted_frame, offset_x, offset_y)
            
            # Display the clipped image
            cv2.imshow('2D Mode', clipped_frame)
            
        elif flag == "3d":
            # Set offsets for left and right eye
            offset_left_x, offset_left_y = -5, 150  # Example: for left eye
            offset_right_x, offset_right_y = 18, 150  # Example: for right eye
            
            # Clip and combine images
            combined_frame = clip_and_combine_image(undistorted_frame, offset_left_x, offset_left_y, offset_right_x, offset_right_y)
            
            # Display the combined image
            cv2.imshow('3D Mode', combined_frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
