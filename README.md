# Age-and-Gender-Detection-Deep-Neural-Network
Age-and-Gender-Detection-Deep-Neural-Network
## Developed by students Huda Zakir Khan	Aparna Suresh with my supervision 
# Age-and-Gender-Detection
This project detects age and gender with the help of facial features using OpenCV.

## How To
We created the program using Jupyter Notebook. In order to run the program, first download the models mentioned in the ‘Age and Gender Detection - Method and Techniques’ document onto your device. Save them in a folder and import it into the Jupyter Notebook.\
The packages needed to run the program are cv2, and os. Before importing the cv2 package, we need to install open cv, which is done using the below code.
pip install opencv-python\
OpenCV is a software program that focuses on image processing, video analysis, or computer vision. When it comes to analyzing photos and videos using advanced digital algorithms, OpenCV can assist developers tackle a variety of difficulties in different sector. The module includes strategies that are commonly used when, for example, recognizing faces. After receiving photos or videos as input, the module applies filters that convert them to boolean values, allowing features to be recognized using comparison functions based on shared characteristics.\
To take a live video from your camera, run the program as it is. To take an image as input, comment those statements which have a comment ‘to read video’, and uncomment the statements which say ‘to read image’.

## Methods and Techniques
We use the following models in our code. These models contain a convolutional neural network for face detection, age detection, and gender detection. These models are trained using the Adience Benchmark Datasets. \
●	opencv_face_detector.pbtxt\
●	opencv_face_detector_uint8.pb\
●	age_deploy.prototxt\
●	age_net.caffemodel\
●	gender_deploy.prototxt\
●	gender_net.caffemodel\
We use both live video and a picture as input. The picture undergoes image preprocessing before being introduced into the trained model, as the model is trained for a certain size, scale, alignment of image. After preprocessing the image runs through each of the networks for face, age, and gender detection. Only three convolutional layers and two fully-connected layers with a modest number of neurons make up the age and gender network. \
The first two convolution layers are followed by a rectified linear operator (ReLU), a max-pooling layer, and a local response normalization layer each. The third convolution layer is followed by a rectified linear operator (ReLU) and a max-pooling layer. The first two fully-connected layers are followed by ReLu and a dropout layer. A soft-max layer receives the output of the last fully connected layer, which assigns a probability to each class. The prediction is made by selecting the class with the highest probability for the test image in question.\
For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers. Both the caffe models (age_net and gender_net) simply define the internal states of the parameters/gradients of those layers. \

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
