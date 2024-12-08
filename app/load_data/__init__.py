
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

def loadFrame( save_path="frame1.png"):


    # Load the images back using matplotlib and convert them to uint8
    frame_rgb = plt.imread(save_path)
    if np.max(frame_rgb) <= 1:# if the image is in the range [0,1]
        frame_rgb = (frame_rgb*255).astype(np.uint8)
    return frame_rgb




def saveVideoFrames(video_path,save_path):
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
        # Convert the frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Save the frames as image files
        count+=1
        cv.imwrite(f'{save_path}\\{count}.png', frame_rgb)  
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