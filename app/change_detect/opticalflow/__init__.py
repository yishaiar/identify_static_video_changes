import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

def opticalFlow(baseModel, next,magnitude_thresholding = True):
    '''
    input:  prvs - grayscale frame 1, range [0, 255]
            next - grayscale frame 2, range [0, 255]
    output: OpticalFlow_rgb - optical flow visualization in RGB format, range [0, 255]
            OpticalFlow_gray - optical flow visualization in grayscale format, range [0, 255]
    '''
    
    # Compute optical flow between the prvs and next frames; prvs and next are grayscale frames
#     there are two types of optical flow:

#     1. Lucas-Kanade Optical Flow - Sparse Optical Flow (Estimates motion for a few points in the frame); Less sensitive to noise
#     prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
#     curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

#     2. Farneback Optical Flow - Dense Optical Flow (Estimates motion for every pixel in the frame); Sensitive to noise in low-resolution videos
    flow = cv.calcOpticalFlowFarneback(baseModel, next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # Convert the flow to magnitude and angle
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    
    if magnitude_thresholding:
        mag_threshold=(0.5, 10.0)
        min_thresh, max_thresh = mag_threshold
        valid_mask = (mag > min_thresh) & (mag < max_thresh)

        # Mask the flow based on the threshold
        flow[..., 0] = flow[..., 0] * valid_mask
        flow[..., 1] = flow[..., 1] * valid_mask
        mag = mag * valid_mask
        ang = ang * valid_mask


    # Visualize the flow in HSV format
    hsv = np.zeros((baseModel.shape[0], baseModel.shape[1], 3),dtype='uint8')
    hsv[..., 1] = 255               # Full saturation (maximum) i.e full color intensity.
    hsv[..., 0] = ang * 180 / np.pi / 2 # Direction in the range [0, 180]
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # Normalized magnitude in the range [0, 255]

    # Convert the HSV image to BGR for display
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    OpticalFlow_rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    OpticalFlow_gray = cv.cvtColor(OpticalFlow_rgb, cv.COLOR_RGB2GRAY)
    return OpticalFlow_rgb,OpticalFlow_gray