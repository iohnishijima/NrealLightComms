import numpy as np
import cv2
import glob

# Chessboard settings (may need to be adjusted)
chessboard_size = (9, 6)  # Number of internal corners of the chessboard (width, height)

# Lists to store corner points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare the real world coordinates for chessboard corners (starting from (0,0,0))
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Load images
images = glob.glob('Path to your images folder/*.jpg')  # Specify the path to your images

for fname in images:
    print(f"open Image: {fname}")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If corners are found, add points to list
    if ret:
        print(f"find corners at: {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)

print("started calculation")
# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# Display calibration results
print("Camera matrix:", camera_matrix)
print("Distortion coefficients:", dist_coeffs)


img = cv2.imread('Path to your test image.jpg')
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
