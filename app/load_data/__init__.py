
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def loadFrame( save_path="frame1.png"):
    '''
    Load images from file using matplotlib and convert them into range [0,255] using uint8
    
    Args:
        save_path (str): The path to the image file.
    Returns:
        np.array: The loaded image in RGB format with range [0, 255].
    '''
    frame_rgb = plt.imread(save_path)
    if np.max(frame_rgb) <= 1:# if the image is in the range [0,1]
        frame_rgb = (frame_rgb*255).astype(np.uint8)
    return frame_rgb





def pixelize_frame(frame, resolution_decrease=10):
    """
    Pixelizes a given frame by resizing it to a lower resolution and then resizing it back.
    
    Args:
        frame (np.array): The input video frame to pixelize. an rgb image with range [0,255]
        pixel_size (int): The size of the "pixels" in the pixelized output.
    
    Returns:
        np.array: The pixelized video frame. an rgb image with range [0,255]
    """
    # Get the current frame size
    height, width = frame.shape[:2]
    
    # Resize the frame to a smaller resolution to simulate the low-res camera
    small_frame = cv.resize(frame, (width // resolution_decrease, height // resolution_decrease), interpolation=cv.INTER_LINEAR)
    
    # Resize back to the original size to get the pixelated effect
    pixelized_frame = cv.resize(small_frame, (width, height), interpolation=cv.INTER_NEAREST)
    return pixelized_frame

def saveVideoFrames(video_path,save_path,pixelize = False,resolution_decrease = 10):
    """
    Saves frames from a video as images, with an option to pixelize them.
    
    Args:
        video_path (str): The path to the input video file.
        save_path (str): The path to save the extracted frames.
        pixelize (bool): Whether to apply the pixelization effect.
        resolution_decrease (int): The amount by which resolution is decreased if pixelization is applied.
    """
    # Open the video file
    cap = cv.VideoCapture(cv.samples.findFile(video_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    count = 0
    while True:
        # read frame by frame
        ret, frame = cap.read()
        # 
        if not ret:
            cap.release()
            break
        
        # Optionally apply pixelization
        if pixelize:
            frame = pixelize_frame(frame, resolution_decrease)
        # Convert the frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Save the frames as image files
        count+=1
        cv.imwrite(f'{save_path}\\{count}.png', frame_rgb)
        
        if count>10:
            break  
    print(f'{count} Frames from video {video_path}')
    print(f'Frames saved in {save_path}')
    
def plotFrames(frame1, frame2,str1='Frame',str2='Base Model'):
    '''
    input:  frame1 - frame in RGB format, range [0, 255]
            frame2 - frame 2 in RGB format, range [0, 255]

    output: None
    '''
    # Create a figure with multiple subplots for each ROI
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the full frames and optical flow result
    for i,[frame,str] in enumerate(zip([frame1,frame2],[str1,str2])):
        axes[i].imshow(frame)  
        axes[i].set_title(str)
        axes[i].axis('off')



    plt.tight_layout()
    plt.show()
    
def saveImagesList(cropped_regions,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(cropped_regions)):
        cv.imwrite(f'{save_path}\\{i}.png', cropped_regions[i])
    print(f'{len(cropped_regions)} ROIs saved in {save_path}')
    
def loadImagesList(save_path):
    imgs = []
    for fname in os.listdir(save_path):
        imgs.append(loadFrame( save_path=f'{save_path}\\{fname}'))
        
        # cv.imshow('img',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    return imgs
    
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # for i in range(len(cropped_regions)):
    #     cv.imwrite(f'{save_path}\\{i}.png', cropped_regions[i])
    # print(f'{len(cropped_regions)} ROIs saved in {save_path}')
    
    
def grayFrame (frame_rgb):
    # Convert the RGB frame to grayscale
    return cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)