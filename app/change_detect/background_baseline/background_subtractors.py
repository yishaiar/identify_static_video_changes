import cv2 as cv, os, numpy as np

# Define the methods and their respective Foreground-Background Subtractors, grades: 1 = Best, 5 = Worst 
methods = {
    "MOG2": cv.createBackgroundSubtractorMOG2(),
        # grade:3, Improved Mixture of Gaussians. Decent for low-resolution, supports shadow detection, but struggles with noise.",
    "KNN": cv.createBackgroundSubtractorKNN(),
        # grade:2, K-Nearest Neighbors-based segmentation. Suitable for low-resolution dynamic scenes, balances speed and accuracy.",
    "CNT": cv.bgsegm.createBackgroundSubtractorCNT(),
        # "grade:5, Simple count of pixel values for efficient and faster background subtraction. Very fast and lightweight, but less accurate in noisy, low-resolution environments.",
    "LSBP": cv.bgsegm.createBackgroundSubtractorLSBP(),
        # grade:1, "Local SVD Binary Pattern-based method. Best for low-resolution, handles noise well and adapts to texture features.",
    "GSOC": cv.bgsegm.createBackgroundSubtractorGSOC(),
    #     "grade": 2, "Geometric Stable Optimal Components method. Robust in low-resolution, handles complex background variations well.",
    "PBAS": cv.bgsegm.createBackgroundSubtractorGSOC(),  # Alias for GSOC
        # "grade": 2, "Pixel-Based Adaptive Segmenter (alias for GSOC). Equally suitable for low-resolution cameras.",    
}


def apply_bg_subtraction(fgbg, image_files1, LEN=None,substract=False):
    
    LEN = LEN or len(image_files1)
    for image_path in image_files1[:LEN]:
    
        
        # Read the image files
        frame = cv.imread(image_path)
        
        # Verify the frame was successfully loaded
        if frame is not None:
            
            # Apply the background subtractor to learn the background
            if not substract:
                fgbg.apply(frame)
            
            # Apply the background subtractor to detect foreground (subtract background)
            else:
                fgmask = fgbg.apply(frame)

                # Display the frame and foreground mask
                display_resized_frame(fgmask,frame,image_fname = image_path.split("/")[-1])
                # Wait for 30 ms, and exit if the 'Esc' key is pressed
                k = cv.waitKey(30) & 0xFF
                if k == 27:  # Esc key to exit
                    break

    return fgbg

def display_resized_frame(fgmask,frame,image_fname = ''):    
    # Resize the mask to the same size as the original frame for display purposes (optional)
    fgmask_resized = cv.resize(fgmask, (frame.shape[1], frame.shape[0]))
    
    # Add the filename text to the frame
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, f'{image_fname}', (10, 30), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    
    # Concatenate the frame, foreground mask, and filename horizontally (side by side)
    side_by_side = np.hstack((frame, cv.cvtColor(fgmask_resized, cv.COLOR_GRAY2BGR)))  # Convert fgmask to BGR for display
    
    # Display the concatenated image
    cv.imshow('Frame and Foreground Mask', side_by_side)

def save_background_model(method_name, fgbg, save_dir):
    try:
        # Save the background model as an image
        im = fgbg.getBackgroundImage()
        save_path = os.path.join(save_dir, f'background_model_{method_name}.png')  # Or .jpg if preferred
        cv.imwrite(save_path, im) # uint8 [0,255]
        print(f"{method_name} background model saved as {save_path}")
        
        # cv2.imshow(method_name, im)
        # k = cv2.waitKey(30) & 0xFF

        return True
        
    except Exception as e:
        print(f"Error saving {method_name} background model:")
        print(e)
        return False
    





if __name__ == "__main__":
    pass

