===================
Neural Network for MNIST Digit Classification
===================

This document contains details on how to run the code within this repository, and links the PDF which contains the responses to the questions in this assignment.

Running the Code
-------------------
Anyone can run any of code in this document by running the following commands.

``python setup.py install``

The setup script should install all necessary python libraries to run the code. I am currently running ``Python 3.4.4 :: Anaconda 2.3.0 (x86_64)``.

One can then running the following commands (in the normal Python way).

For digits scripts:

:code:`cd code`

:code:`python script_digits.py`

Configuration
-------------------
This script takes a very long time to run if you run the six-fold cross validation logic inside to completion (between 20 and 40 minutes of training twelve times for the two loss functions!). However, I've included a fair number of constant ("environment") variables at the top of the file in order to make running the script simpler and more efficient. They are described below.

``RUN_XOR``:

``True`` - test the neural network on XOR.

``False`` - skip this integration test.


``RUN_CROSS_VALIDATION``:

``True`` - perform six-fold cross validation on this run of the script.

``False`` - do not cross-validate when running the script.


``RUN_MSE``:

``True`` - perform cross-validation using the MSE loss classifier - only works when ``RUN_CROSS_VALIDATION = True``.

``False`` - do not cross-validate using the MSE loss classifier.


``RUN_CROSS_ENTROPY``:

``True`` - perform cross-validation using the cross-entropy loss classifier - only works when ``RUN_CROSS_VALIDATION = True``.

``False`` - do not cross-validate using the cross-entropy loss classifier.


``RUN_KAGGLE``:

``True`` - train the neural network on all 60000 images and classify the test set.

``False`` - do not train and classify the test set.


``PLOTS``:

``True`` - generate plots of the images (as a sanity check), and generate the plots for submission.

``False`` - skip plot generation (recommended, as they halt the program until the user exits out of each plot).
