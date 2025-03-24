import cv2
import numpy as np
import glob

# Define checkerboard dimensions
CHECKERBOARD = (10, 7)  # (Columns, Rows)

# Prepare object points (3D coordinates of the corners in the real world)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Lists to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all images
images = glob.glob('/home/kevinbee/Desktop/Indoor-Map-GUIs/src/camera_calibration/calibration_images/*.jpg')  # Change path if needed
print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)
    else:
        print(f"Checkerboard not found in: {fname}")

cv2.destroyAllWindows()

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# # Save to file
# np.save("camera_matrix.npy", camera_matrix)
# np.save("dist_coeffs.npy", dist_coeffs)