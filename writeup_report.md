**Behavioral Cloning Project**


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* normal_track.mp4 a video of one lap of the normal track
* mountain_track.mp4 a video of one lap of the mountain track

[//]: # (Image References)

[image1]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/normal_tack_drive.jpg 'Normal Track Center'
[image2]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/mountain_track_drive.jpg 'Mountain Track Right
[image3]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/center.jpg 'Center Camera'
[image4]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/left.jpg 'Left Camera'
[image5]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/right.jpg 'Right Camera'
[image6]: https://github.com/Nervehurter/Behavioral-Cloning/blob/master/examples/center_cropped.jpg 'Center Image Cropped'

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

On the normal track the car keeps to the center of the track and on the mountain track it keeps mostly on tie right lane, only in some sharp corners it briefly crosses the dashed center line. It never drives across the solid lines at the edges of the road though. The following images show scenes from the autonomous driving mode.

![alt text][image1] ![alt text][image2]

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network inspired from NVIDIAS solution for a end-to-end CNN that predicts steering angles from camera images. It uses three convolutional layers with 5x5 and two with 3x3 filter sizes and depths between 24 and 64 (model.py lines 85-95).
After those there are in total four fully connected layers to get the linear steering angle from the last convolutions output.

The model includes RELU layers to introduce nonlinearity (code lines 85-87), and the data is normalized in the model using a Keras lambda layer (code line 82). Also the inpus are cropped to get rid of the sky and far away landscape as well as the hood of the vehicle (code line 83). The following images show the cropping.

![alt text][image3] ![alt text][image6]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 98-100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering. To have an even distribution of left and right turns I recorded forward, as well as reverse laps.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA implementation because they already proved this architecture is capable of predicting correct steering angles. The model has around 250.000 parameters, which might be a little on the heavy side for this simulator but it seems to work well as there was no bad overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more training data with turns, after that it drove nicely.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is three convolutions with 5x5 filter sizes, strides of 2 and relu activation. Then two convolutions with 3x3 filter sizes strides of 1 and relu activation. After flattening there are three fully connected layers with outputs of 100, 50 and 10 and a final output layer that returns a single steering angle value. the dense layers all have linear activation.

#### 3. Creation of the Training Set & Training Process

To capture the correct behavior I recorded forward and reverse laps, as well as some recovery maneuvers. also I recorded a lap of the challenge track, which presumably helps with sharper corners. I was worried that since on the challenge track I drove on the right lane this would also influence the cars behaviour on the normal track, but it worked. Probably the network is able to recognize the type of track.

After the collection process, I had 5.200 number of data points. I then preprocessed this data by extracting the left and right images and adding an offset of +-0.15 to the current steering angle. I found this value to work well compared to 0.1 and up to 0.25. the following pictures show left, center and right image of the same scene.

![alt text][image4] ![alt text][image3] ![alt text][image5]

I finally randomly shuffled the data set and put 10% of the data into a validation set, since the dataset is rather large.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the only minor improvements after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
