# Behavioral Cloning

[Low Res Video of Car Self Driving Around the Track](https://www.youtube.com/watch?v=KUvZUn31UsY)
[//]: # (Image References)

[image1]: readme_images/ackermann.png "Ackermann"
[image2]: readme_images/nvidia.png "Nvidia"



The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report 


My project includes the following files:
* clone.py contains the script to create and train the model
* model.h5 contains the trained convolution neural network from clone.py
* drive.py is for driving the car in autonomous mode (Provided by Udacity)

### Model Architecture and Training Strategy

I replicated the Model Nvidia used in their End to End self driving car.

| Layer         		|     Description	        										| 
|:---------------------:|:-----------------------------------------------------------------:| 
| Input         		| 160x320x3 RGB image  												|
| Pre-processing 		| Normalized -> Flip Image		 									|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU, 2x2 Max Pool, outputs 78x158x24  |
| Convolution 5x5     	| 1x1 stride, valid padding, RELU, 2x2 Max Pool, outputs 38x77x36   |
| Convolution 5x5     	| 1x1 stride, valid padding, RELU, 2x2 Max Pool, outputs 17x36x48   |
| Convolution 3x3     	| 1x1 stride, valid padding, RELU, outputs 15x34x64   				|
| Convolution 3x3     	| 1x1 stride, valid padding, RELU, outputs 13x32x64   				|
| Flatten				| 13x32x64 -> 26624													|
| Fully Connected 		| outputs 100														|
| Fully Connected 		| outputs 50														|
| Fully Connected 		| outputs 10														|
| Fully Connected 		| outputs 1															|

!["Nvidia Model"][image2]


I used adam as the optimizer and mse as the cost function.

Twenty percent of the data was used as validation, and while the validation data of the network is low, and the car can autonomously drive around the track, it doesn't mean too much. The number one rule is that you aren't allowed to let your model see your validation and test data. The model saw a very close approximate of it's validation data in the training data, this is because the data was captured from a video file, and frames between a video are often very similar. The model also saw the test data (the requirement to drive autonomously around a track), because the track that the training data was gathered from the same track it was tested from.

That said, something was learned, but one cant be confident the model would work on a different track.

Dropout and L2 regression wasn't needed, although I did have to reduce the number of epochs from Keras' default of ten to three. I believe because I don't have a very large amount of data, and that the problem is sufficiently complex, the model didn't get a chance to overfit. 

### Training Data

The Data used to train the model came from me manually driving the car around the track four times, while Udacity's software recorded my input. When I stopped the recording, my actions would be replayed from the start and images would be captured along the way. The training data focused on keeping the car in the middle of the road to give it the best chance of success. This posed a minor problem. Because I, fairly accurately, drove in the middle of the road along the track, the model was given little data on how to recover if the car drifts to the side of the road. This was solved by the recording software giving me a left, center, and right camera on the car. Using the left and right images along with a correction value for the steering I was able to simulate me driving without having to spend much time collecting data.

When using the images from the left / right cameras, I used a static correction of +/- .3 (which translates to 7.5 degrees). I believe that this caused my car to have a very wavy path. When driving around a corner the car preformed as expected, but when driving on a straight road the car would over correct and almost swerve left and right. I believe this is because the correction of 7.5 degrees was too much for the straight paths. Although, when I reduced the static correction, the car would under steer on curves and drive off the road. 

The obvious solution is to make the left / right correction dynamic, and to do that one would need to calculate the circle the center camera is traveling in, and then calculate a circle for the left / right camera such that it intersects with the center circle at certain distance. The circle of the center camera depends on too many real world factors to calculate such as, wind speed, road traction, front/back weight balance, etc...

That said a basic calculation of the center camera circle can be done by extending a line perpendicular from the inside wheel's lines and using where they connect as the center of the circle.

!['Ackermann Exmaple'][image1]
*Note the front outside wheel would only intersect with the inside of the circle if the car is using Ackermann steering.

Once the center camera circle has been calculated, one could shift a new the circle to the left or right equal to the distance between the center and outside cameras, calculate the radius increase / decrease required to intersect with the center line at a distance x, and then calculate the new required angle of the steering. This method would require knowing the distance between the center and outside cameras, and the distance between the front axle and rear axle. Neither of these were provided from the simulation but it would be trivial to get these from a physical car. 

### Solution Design Approach

The overall strategy for deriving a model architecture was to tinker with convolutional networks and try to use different well known convolutional architectures . First I tried a one layer convolutional network and it could stay in line on a straight shot. From there I used LeNet and it could do some curves. Finally I used Nvidia's architecture which was able to drive around a track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Creation of the Training Set & Training Process

As mentioned before, I drove 4 laps around the first track. I then augmented the data to utilize the left and right cameras, as well as flipped the images. This provided me with 6x my initial data. Had that not been enough I was considering driving backwards around the first track as well as flipping the images upside down in hopes of really hammering in what the road was. 


After the collection process, I had around 56000 images to train my network on. Beyond data augmentation, for preprocessing I just normalized the data. 
