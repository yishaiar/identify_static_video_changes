
import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def cropROI(roi_coords,image ):
    # Crop the region from the original image
    cropped_regions = []
    for roi in roi_coords:
        y1, x1, y2, x2 = roi
        cropped_regions.append(image[y1:y2, x1:x2]) # (top_left_y, top_left_x, bottom_right_y, bottom_right_x)
    return cropped_regions

def detectROIs(thresh,image, min_area=2000, max_area=20000):
    '''
    Detect regions of interest (ROIs) based on contours in the given image.
    The function assumes that the regions are distinguishable based on intensity.

    input:  image - The image in which to detect the regions (grayscale or optical flow result)
            min_area - Minimum area of contours to consider as regions (default 1000 pixels)
            max_area - Maximum area of contours to consider as regions (default 10000 pixels)
    output: roi_coords - List of tuples containing (y1, x1, y2, x2) coordinates for each ROI
            cropped_regions - List of cropped images corresponding to each detected region
    '''
    
    # # Step 1: Convert the image to grayscale if it is not already
    # if len(image.shape) == 3:
    #     gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # else:
    #     gray = image
    
    # # Step 2: Threshold the image to get a binary mask (regions of interest)
    # _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    
    # Step 3: Find contours in the thresholded image
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    

    # Step 4: Filter contours based on area and extract bounding boxes
    roi_coords = []
    for contour in contours:
        area = cv.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        # Get bounding box (x, y, width, height)
        x, y, w, h = cv.boundingRect(contour)
        roi_coords.append((y, x, y + h, x + w))  # (top_left_y, top_left_x, bottom_right_y, bottom_right_x)
        
    
    
    return roi_coords


# roi_coords, cropped_regions = detectAndCropROIs(OpticalFlow_rgb)



# # ---------------------------

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def adaptiveThresh(image, thresh_type = 0):
    # Step 1: Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if thresh_type == 0:
        # Step 2: Threshold the image to get a binary mask (regions of interest)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    else:
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Threshold the image using adaptive thresholding
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv.THRESH_BINARY, 11, 2)
    
    

    
    # # --- Debug: Show the thresholded image ---
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(thresh, cmap='gray')
    # plt.title('Thresholded Image')
    # plt.axis('off')
    
    return thresh


def plotWithROIs(roi_coords,baseModel_rgb, next_rgb, OpticalFlow_rgb = None ):
    '''
    input:  prvs_rgb - frame 1 in RGB format, range [0, 255]
            next_rgb - frame 2 in RGB format, range [0, 255]
            OpticalFlow_rgb - optical flow visualization in RGB format, range [0, 255]
            OpticalFlow_gray - optical flow visualization in grayscale format, range [0, 255]
            roi_coords - List of tuples containing (y1, x1, y2, x2) coordinates for each detected ROI
            cropped_regions - List of cropped images corresponding to each detected region
    output: None
    '''

    
    # Plot the full frames and optical flow result
    

    imgs = [next_rgb, baseModel_rgb, ] if OpticalFlow_rgb is None else [next_rgb,baseModel_rgb,  OpticalFlow_rgb,cv.cvtColor(OpticalFlow_rgb, cv.COLOR_RGB2GRAY)]
    titles = ['Frame','Base Model'] if OpticalFlow_rgb is None else ['Frame','Base Model','Optical Flow','Optical Flow - Grayscale']

    # Create a figure with multiple subplots for each ROI
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i, [img,title] in enumerate(zip(imgs,titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    
    # Draw rectangles around the regions of interest in the OpticalFlow_rgb image
    for i in range(len(axes)):
        for idx, (coords) in enumerate(roi_coords):
            y1, x1, y2, x2 = coords
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            axes[i].add_patch(rect)  # Adding rectangle to OpticalFlow image
    plt.tight_layout()
    plt.show()
    
def plotCrop(cropped_regions):
    
    fig, axes = plt.subplots(len(cropped_regions), 1, figsize=(5, int(len(cropped_regions)*3)))

    # Plot cropped regions
    for idx, cropped in enumerate(cropped_regions):
        
        axes[idx].imshow(cropped)
        axes[idx].set_title(f'Cropped Region {idx + 1}')
        axes[idx].axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()