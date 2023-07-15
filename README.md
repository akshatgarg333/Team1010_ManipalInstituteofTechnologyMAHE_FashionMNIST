# IntelUnnati_Team1010
Hi, to everyone viewing this project . Fashion MNIST is a classification problem in which images of size 28X28 are classified into 10 categories. For this project convolutional neural network are used which are complex and time taking. So, to test why normal nn are not used for this, I first tested using a tri layer nn model which did the work but it was sacrificing a lot on the accuracy side. Then I started to build a CNN model using tensorflow which after many days of changes and small tweakings ended with a model of 20 layers. This was the best model created with the highest accuracy. Then I also tried to use pyTorch which was very similar to the tensorFlow model. The Pytorch model had less accuracy than tensorflow. Even after applying the IPEX( Intel Extension for Pytorch) the accuracy boosted but not my much. Hence, the final model I used was the TensorFlow cnn model which when optimised using the openvino platform had reduced inference time.I have attached the project report along and the videos for each code are there inside each of the folder.

If the videos are not opening from github I have uploaded them as an unlisted video on youtube, the links for them can be found below

demo_video1=https://youtu.be/y1zRJytrwsI \
demo_video2=https://youtu.be/njvnVr34bjM \
demo_video3=https://www.youtube.com/watch?v=nqVdI0ohxCo
