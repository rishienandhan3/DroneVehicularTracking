## Drone follow me using Kalman Filters

The codebase for 'Drone follow me using Kalman Filters' assignment-3 is present in rb5291_assignment_3.ipynb jupyter notebook. The training part of this pipeline was performed in Colab Pro using V100 GPU and inference was run on local computer running M3 Pro Apple Silicon using GPU support.

The predictions and state estimates as output video file are present in the shared drive which can be accessed using this link: https://drive.google.com/drive/folders/1Bd1D3jyy9EzcCr2YJHxMZ4rRgvGimuCh?usp=sharing.

- We saved the videos with predictions using YOLOv8m (medium size) pretrained on COCO dataset to 'outputs/COCO_pretrain/' folder.
- We saved the videos with predictions using YOLOv8m (medium size) pretrained on COCO and trained on VisDrone to 'outputs/VisDrone_train_results/' folder.
- We saved the final prediction video files with kalman filters in 'outputs/Final_predictions/' folder.
- The 'outputs/VisDrone_train_plots/' folder in the shared drive contains the training results such as confusion matrix, loss and acuracy plots.
- 'outputs/Model_Weights/' folder contains the model weights of the final predictions model.

Within the notebook, you'll find well-documented code along with detailed explanations of the process, explanation of codes written & APIs used, instructions on how to run the code, and discussions on the results and outputs. However, copying over some explanations to this readme file. Note that using the Jupyter notebook reader on Github causes the entire cell output to be visible, hence viewing this file on local computer might help truncate the long detection outputs from certain cells. 

## Task 1: Video Library
We use Pytube python library for downloading the videos from YouTube. Pytube is a small, dependency-free Python module for accessing videos from the internet. In this use-case, we only download the videos and not the captions as we do not seem to use captions in this assignment.

Official Docs: https://pytube.io/en/latest/user/quickstart.html

## Task 2: Object Detection
We use the Ultralytics API to instantiate a YOLOv8m (medium size version of YOLOv8) object detection model, pretrained on COCO dataset.

For small objects since we use drone imaging, lower-stride models typically fare better as these models generally maintain more detail from the input image, which can be essential for detecting and correctly classifying small objects. Hence we chose to use YOLOv8m model for our purposes. This medium size model also does not have extremely large number of parameters, to run inference using local system.

#### Visualize predictions on local computer

We write code that utilizes YOLOv8 pre-trained on COCO dataset for object tracking on a specified input video stream.

The object detection tracking results are visualized by annotating the frame with bounding boxes, labels and confidence scores for detected objects. The annotated frame is then displayed in a separate window.

#### Save the Video file with predictions using YOLOv8 (medium) pre-trained on COCO dataset

We then save the result of object detection on our three input videos using the YOLOv8 model pre-trained on COCO, we use this model as the baseline. We make further improvement to this object detection model in the next sub-section.

We saved the videos with predictions using YOLOv8m (medium size) pretrained on COCO dataset to 'outputs/COCO_pretrain/' folder to a shared drive which can be accessed using this link: https://drive.google.com/drive/folders/1Bd1D3jyy9EzcCr2YJHxMZ4rRgvGimuCh?usp=sharing.

Notice how in video1.mp4, there are multiple erroneous detections such as cake, boat, knife, train, clock, along with irregular detection of cars, being our object of interest. Also, notice how the detection of the car in video1 is fluctuating across frames; this can be attributed to the object being extremely small in the Drone imagery and hence the object detector fails to constantly pick it up across frames. The same pattern can be observed in output videos video2.mp4, video3.mp4 as well.

So, as an imporvement to our object detection model, we now train it on the VisDrone dataset.


#### Train YOLOv8m on VisDrone dataset

[Bonus Points Question] We now train our YOLOv8m model on VisDrone dataset, which contains objects from 10 classes inlcuding bicycle, pedestrians, car, truck, etc. There are multiple detections across 6471 images that we use for training, 548 images for validation, and 1610 images for test purposes. Each image contains multiple detections across various objects from 10 classes. 

The training part of this pipeline was performed in Colab Pro using V100 GPU, using an image size of 640x640 for 100 epochs. The total training took ~8hours using this GPU. We have used a YAML file to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. The VisDrone.yaml file present in the parent directory is maintained for this purpose. We use the help of the Ultralytics API for training our model. 

The training results across 100 epochs are present in the cell outputs in the notebook.

The 'VisDrone_train_plots' folder in the shared drive contains the training results such as confusion matrix, loss and acuracy plots. The results.csv file in this folder contains various training metrics such as box loss, classification loss, precision, recall, and other metrics for train and validation datasets for every epoch. 

Now, we use the YOLOv8m model pre-trained on COCO and trained on VisDrone dataset to run predictions on our input video streams. 

#### Save the Video file with predictions using YOLOv8 (medium) pre-trained on COCO dataset and trained on VisDrone for 100 epochs

We now save the videos with predictions using YOLOv8m (medium size) trained on VisDrone to 'outputs/VisDrone_train_results/' folder in the shared drive which can be accessed using this link: https://drive.google.com/drive/folders/1Bd1D3jyy9EzcCr2YJHxMZ4rRgvGimuCh?usp=sharing.

With our newly trained model, we are able to avoid erroneous detections and more importantly, pick up important features (car and people) more constantly across our video frames. You can observe in video1.mp4 that the detection of the car is more stable and is being picked up constantly across frames, which would certainly help us in tracking in next step. 

The same pattern is observed in other output videos as well. Hence, training our model on VisDrone dataset certainly helps in a more stable and continuous object tracking. 


## Task 3: Kalman Filters
The write function that creates and initializes a Kalman Filter model along with its parameters for 2D motion object tracking model with specified dimensions.

We can then integrate this Kalman Filter model into the object tracking pipeline along with the object detections obtained from Task 2 to track the pedestrians and vehicles in the videos. We can use the predicted object positions from the Kalman Filter to plot trajectories and improve tracking accuracy. We plan to detect and track cars and people across video frames. 

We integrate the Kalman Filter into our object detection pipeline to estimate state of objects in subsequent frames.
Following are the steps we follow:

- Initialize the YOLO model with pre-trained weights from VisDrone dataset training.
- Create dictionaries to store Kalman filters (`kalman_filters`) for each object ID and past trajectories (`trajectories`) for each object ID.
- Define the number of past frames to keep track of (`num_past_frames`).
- Enter a loop to process each frame of the video until the end of the video is reached or the user presses 'q' to exit.
- Perform object detection and tracking on the frame using the YOLO model.
- Extract bounding boxes, object IDs, and predicted classes from the detection results.
- Iterate over each detected object in a given frame and update its Kalman filter to predict its next position (St+1).
- Store the current position in the trajectory list of past estimates for each object to draw trajectory.
- Visualize the object trajectory and bounding box on the frame by drawing lines and rectangles.
- Annotate the frame with the object class and ID using `cv2.putText`.
- Terminate the loop and release the video capture object when the 'q' key is pressed.

The inline comments in the code provide additional details.

We save the predictions and state estimates as a video file in 'outputs/Final_predictions/' folder in the shared drive which can be accessed using this link: https://drive.google.com/drive/folders/1Bd1D3jyy9EzcCr2YJHxMZ4rRgvGimuCh?usp=sharing.


In each video outputs, the green point is the predicted estimate of the state of object (centroid of detected object) in next frame(St+1). The red rectangle is the bounding box predicted using the object detector and the blue trailing points represents the past state estimates of an object from its past 10 frames, which we call trajectory. 


**NOTE:** Notice how the blue dots represents the past state estimates of an object from its past 10 frames, but it is important to note this is not very indicative of its actual trajectory as the camera also moves relative to the object causing the blue points to be extremely wavery. Since the object follows a motion model and our camera also moves, the trajectory points (blue) may not be realistic.

What is important to note is our green dot (centroid of the object), which gives the estimate of the next state of the object (St+1) in the next time step (next frame). We obtain the estimate of the next immediate state of the object as this green dot, successfully through the recursive state estimate model (RSE) using Kalman Filters.


Each object detected in the video frames have a unique ID maintained, which helps track multiple objects simultaneously in a video frame. There is a unique bounding box and state estimate for every unique object detected across frames. Note from the code that we maintain  dictionary to store Kalman Filters for each object ID, so have one Kalman filter to track each of the required and present objects (cars and people).


#### To address false positives in object detection, we can use some of the following methods:
1. **Temporal Consistency with Tracking:** The tracker, especially when using methods like Kalman filters, provides temporal consistency by predicting the position of objects in subsequent frames based on their previous positions. By leveraging this temporal information, we can verify the consistency of detections over time. If a detection is inconsistent with the predicted trajectory, it is more likely to be a false positive and can be discarded.

1. Dynamically Adjusting Detection Threshold: Object detection such as YOLOv8 produce detection scores along with bounding boxes. By adjusting the detection threshold for each class individually based on the precision required, we can filter out detections with low confidence scores dynamically.

3. Non-Maximum Suppression (NMS): NMS is a technique used to suppress multiple overlapping bounding boxes for the same object. It retains only the bounding box with the highest confidence score while suppressing others. This helps in removing redundant detections and reducing false positives.
