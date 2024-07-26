import os

import cv2
import numpy as np


def preprocess_image(image, target_size=None):
    if image is None:
        raise ValueError("Image not loaded correctly. Check the file path.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if target_size:
        gray_image = cv2.resize(gray_image, target_size)
    return gray_image


def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def find_piece_position(full_image, piece_image, full_size=(1000, 1000), piece_size=(100, 100)):
    # Preprocess images
    try:
        full_image = preprocess_image(full_image, full_size)
        piece_image = preprocess_image(piece_image, piece_size)
    except ValueError as e:
        print(e)
        return None

    # Extract features
    kp1, des1 = extract_features(full_image)
    kp2, des2 = extract_features(piece_image)

    # Match features
    matches = match_features(des1, des2)

    if len(matches) == 0:
        return None

    # Extract location of good matches
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return None

    # Use homography to find the position in the full image
    h, w = piece_image.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    return dst

# os.system("pwd")
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file at path {image_path} does not exist.")

    image = cv2.cv.LoadImage(image_path, cv2.CV_LOAD_IMAGE_COLOR)
    if image is None:
        raise IOError(f"Failed to load image from path {image_path}. Check if the file is corrupted or in an unsupported format.")

    return image

# full_image = load_image('/Users/arminforoughi/Documents/solvejigsaw/src/IMG_9699.JPEG')

full_image = cv2.imread('/Users/arminforoughi/Documents/solvejigsaw/src/IMG_9699.jpg')
# full_image  cv2.imread('/Users/arminforoughi/Documents/solvejigsaw/src/IMG_9699.jpg')



# cv2.imshow("", full_image)
#
# cv2.waitKey(0)



piece_image = cv2.imread('IMG_9700.jpg')
print(full_image)



# position = find_piece_position(full_image, piece_image)
#
# if position is not None:
#     print("Piece position in the puzzle image:")
#     print(position)
# else:
#     print("Piece could not be matched.")
