# Historical-arabic-manuscripts-word-spotting
Word spotting for historical Arabic manuscripts using convolutional neural network

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
