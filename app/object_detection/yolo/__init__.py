from torch import save,load,hub
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def saveYOLOv5Model(data_dir):
    # Save model weights to a specific path
    save_path = f'{data_dir}/yolov5s_weights.pt'
    if os.path.exists(save_path):
        print(f"Model weights already saved to {save_path}")
    else:
        # Load the model
        model = hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Save model weights to a specific path
        save(model, save_path)
        os.remove('yolov5s.pt')
        print(f"Model weights saved to {save_path}")
    


def saveYOLOv8Model(data_dir):
    # Save model weights to a specific path
    save_path = f'{data_dir}/yolov8s_weights.pt'
    if os.path.exists(save_path):
        print(f"Model weights already saved to {save_path}")
    else:
        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8s.pt')
        # delete file yolov8s.pt
        
        model.save(save_path)
        os.remove('yolov8s.pt')
        print(f"Model weights saved to {save_path}")
    
    
# def loadYOLOv5Model(data_dir):

#     # Load the saved weights
#     save_path = f'{data_dir}/yolov5s_weights.pt'
#     if not os.path.exists(save_path):
#         saveYOLOv5Model(data_dir)
#     model = load(save_path)
#     model.eval()  # Set the model to evaluation mode
#     print("Model  YOLO V5 weights loaded successfully")
#     return model

def loadYOLOv8Model(data_dir):
    # Load the saved model weights
    save_path = f'{data_dir}/yolov8s_weights.pt'
    if not os.path.exists(save_path):
        saveYOLOv8Model(data_dir)
    model = YOLO(save_path)
    model.eval()  # Set the model to evaluation mode
    print("Model YOLO V8 weights loaded successfully")
    return model


# def loadModelYOLO(data_dir,model_version = 8):
#     # Load the saved model weights of yolo v8 or v5
    
#     try:
#         save_path = f'{data_dir}/yolov{model_version}s_weights.pt'
#         model = YOLO(save_path) if model_version==8 else load(save_path)
#         model.eval()  # Set the model to evaluation mode
#         print(F"Model YOLO V{model_version} weights loaded successfully")
#         return model
#     except:
#         print(f"error:Model YOLO V{model_version} weights not found in:")
#         print(save_path)
#         return saveYOLOv8Model(data_dir) if model_version==8 else  saveYOLOv5Model(data_dir)


def inferenceYOLOv8Model(model,imgs):
    # Initialize an empty list to store results
    all_results = []

    # Iterate through each image and extract results
    for img in imgs:
        results = model(img)  # Inference results for the image
        
        # Extract bounding boxes and other details
        boxes = results[0].boxes  # Bounding box predictions for the first image
        
        # Add the results for this image to the all_results list
        for i in range(len(boxes)):
            all_results.append({
                # 'image': img,
                # 'xmin': boxes.xyxy[i][0].cpu().numpy(),
                # 'ymin': boxes.xyxy[i][1].cpu().numpy(),
                # 'xmax': boxes.xyxy[i][2].cpu().numpy(),
                # 'ymax': boxes.xyxy[i][3].cpu().numpy(),
                'confidence': boxes.conf[i].cpu().numpy(),
                'class': int(boxes.cls[i].cpu().numpy()),
                'name': model.names[int(boxes.cls[i].cpu().numpy())]
            })
    return all_results

# def inferenceYOLOv5Model(model, imgs):
#     # Initialize an empty list to store results
#     all_results = []

#     # Iterate through each image and extract results
#     for img in imgs:
#         results = model(img)  # Inference results for the image

#         # Extract bounding boxes and other details from the results
#         boxes = results.xyxy[0].cpu().numpy()  # Bounding box predictions (x1, y1, x2, y2)
#         confs = results.conf[0].cpu().numpy()  # Confidence scores
#         cls = results.cls[0].cpu().numpy()  # Class indices

#         # Iterate through the detections and collect information
#         for i in range(len(boxes)):
#             all_results.append({
#                 'confidence': confs[i],
#                 'class': int(cls[i]),
#                 'name': model.names[int(cls[i])],
#                 'xmin': boxes[i][0],
#                 'ymin': boxes[i][1],
#                 'xmax': boxes[i][2],
#                 'ymax': boxes[i][3]
#             })

#     return all_results


def plotClassifcation(roi_coords,next_rgb,roi_detected ):
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
    

    imgs = [next_rgb ]
    titles = ['Frame'] 

    # Create a figure with multiple subplots for each ROI
    fig, axes = plt.subplots(len(imgs), len(imgs), figsize=(15, 5))
    for i, [img,title] in enumerate(zip(imgs,titles)):
        axs = axes if len(imgs) == 1 else axes[i]
        axs.imshow(img)
        axs.set_title(title)
        axs.axis('off')

    
    # Draw rectangles around the regions of interest in the OpticalFlow_rgb image
    for i in range(len(imgs)):
        axs = axes if len(imgs) == 1 else axes[i]
        for idx, (coords) in enumerate(roi_coords):
            y1, x1, y2, x2 = coords
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            axs.add_patch(rect)  # Adding rectangle to OpticalFlow image
            confidence = str(np.round(roi_detected[idx]['confidence'],2))
            axs.text(x1, y1+0.1, f"{roi_detected[idx]['name']}, {confidence}", fontsize=15, color='b', weight='bold')
    plt.tight_layout()
    plt.show()