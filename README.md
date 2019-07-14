# Historical-arabic-manuscripts-word-spotting
Word spotting for historical Arabic manuscripts using convolutional neural network


The following commands are essential in order to run the model with efficiency the first step is installing and running keras-tensorflow:

Upgrade pip:

cd .. //go to the specified directory in python

pip install --upgrade pip

installation of opencv:

pip install openCV-python

install keras:

pip install keras

pip install python-dateutil

start installing tensorflow gpu because it is important to set TensorFlow gpu as back-ground to run the project:

pip install tensorflow-gpu

Run your python file on the specified gpu:

import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


Also another way to run the model is to go to https://colab.research.google.com this is a running environment that the user will be able to adjust a cup or gpu or tpu and has already all the libraries installed, the user can upload the python scripts and dataset and then run them in cells or can also attach them from google drive or provide github re-pository.




In order to run the python files specified in the folders you must follow the following sequence
of files, note the mentioned steps has to be executed in the specified order in order for the project to be executed with no errors:
Install the dataset:

full data set: https://drive.google.com/file/d/1Eg9KEUzPpiWZPQF91lHll0a6fcpuGs2h/view (5GB)

xml ground truth files: https://drive.google.com/file/d/1Cw8CTYxXXbu11JhOF4zSudyEA9RX-rjJ/view

Pruning the images into words and saving them in files (classes):

Run DatasetCrop.py

Renaming the generated classes of words:

Run CalssRenamer.py

Extracting the files that will be used as validating set:

Run validatedata.py

Extracting files that will be used as training set:

Run traindata.py

Extracting files that will be used as testing set:

Run testdata.py

Training the network and creating adjustments in the data training seizes and testing process:

Run model.py
