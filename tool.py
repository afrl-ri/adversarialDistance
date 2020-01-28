##
##   Distribution A. Approved for public release: distribution unlimited.  Case 88ABW-2019-1334. 27 March 2019.
##

## THIS SOFTWARE AND ANY ACCOMPANYING DOCUMENTATION IS    
## RELEASED "AS IS." THE US GOVERNMENT MAKES NO WARRANTY  
## OF ANY KIND, EXPRESS OR IMPLIED, CONCERNING THIS SOFTWARE  
## AND ANY ACCOMPANYING DOCUMENTATION, INCLUDING,  
## WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY   
## OR FITNES FOR A PARTICULAR PURPOSE. IN NO EVENT WILL THE  
## US GOVERNMENT BE LIABLE FOR ANY DAMAGES, INCLUDING ANY  
## LOST PROFITS, LOST SAVINGS, OR OTHER INCIDENTAL OR  
## CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE, OR  
## INABILITY TO USE, THIS SOFTWARE OR ANY ACCOMPANYING   
## DOCUMENTATION, EVEN IF INFORMED IN ADVANCE OF THE  
## POSSIBILITY OF SUCH DAMAGES.    


import keras
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Sequential, Model, Dense, Acivation, Conv2D, MaxPooling2D, Flatten, regularizers, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.metrics.pairwise import euclidean_distances
from keras.datasets import cifar10

from scipy.interpolate import interp1d
import statsmodels.api as sm

import numpy as np
import os
import sys
import random
import math
import foolbox
import pandas as pd
from foolbox.criteria import TargetClass
from foolbox.distances import MAE
from foolbox.criteria import Misclassification
from foolbox.distances import MSE


import tabulate
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Dataset creators                            #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################

## Create and save numpy arrays for the cat dog dataset
##
## @description Creates numpy arrays for the cat dog dataset, assumes structure as downloaded from https://www.kaggle.com/c/dogs-vs-cats/data.
##              Splits into train, validation, and test datasets.
##
## @param rootPath path to the "train" folder
## @param ROW the desired number of rows for each image
## @param COL the desired number of columns for each image
##
## @return saves numpy arrays to the current working directory
##
def catDogDatasetCreator(rootPath, ROW, COL):
    all_images = os.listdir(rootPath)
    dataset = np.ndarray(shape=(len(all_images), ROW, COL, 3), dtype=np.float32)    ## place to save images
    labels = np.zeros(len(all_images))                                              ## place to save labels
    i = 0                                                                           ## assume all images are cats

    for file in all_images:
        img = load_img(rootPath + '/' + file, target_size=(ROW, COL))               ## load images
        dataset[i] = img_to_array(img)
        if 'dog' in file:
            labels[i] = 1                                                           ## mark dogs
        i = i + 1
    
    trainInd = random.sample(range(len(all_images)), math.ceil(2*len(all_images)/3))
    trainImages1 = dataset[trainInd, :, :, :]
    trainLabels1 = labels[trainInd]
    testImages = dataset[[i for i in range(len(all_images)) if i not in trainInd], :, :, :]
    testLabels = labels[[i for i in range(len(all_images)) if i not in trainInd]]
    valInd = random.sample(range(len(trainLabels1)), math.ceil(len(trainLabels1)/10))
    valImages = trainImages1[valInd, :, :, :]
    valLabels = trainLabels1[valInd]
    trainImages = trainImages1[[i for i in range(len(trainImages1)) if i not in valInd], :, :, :]
    trainlabels = trainLabels1[[i for i in range(len(trainImages1)) if i not in valInd]]

    np.save('trainImages', trainImages)
    np.save('testImages', testImages)
    np.save('valImages', valImages)
    np.save('trainLabels', trainlabels)
    np.save('testLabels', testLabels)
    np.save('valLabels', valLabels)

## Create and save numpy arrays for the biased cat dog dataset
##
## @description Creates numpy arrays for the cat dog dataset, assumes structure as downloaded from aiweb.cs.washington.edu/ai/unkunk18.
##              Splits into train, validation, and test datasets.
##
## @param trainCatPath path to the train cat images
## @param trainDogPath path to the train dog images
## @param testCatPath path to the test cat images
## @param testDogPath path to the test dog images
## @param ROW the desired number of rows for each image
## @param COL the desired number of columns for each image
##
## @return saves numpy arrays to the current working directory
##    
def catDogDatasetCreatorBiased(trainCatPath, trainDogPath, testCatPath, testDogPath, ROW, COL):
    
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    
    trainCat = listdir_nohidden(trainCatPath)
    trainCat = [trainCatPath + file for file in trainCat]
    trainDog = listdir_nohidden(trainDogPath)
    trainDog = [trainDogPath + file for file in trainDog]
    testCat = listdir_nohidden(testCatPath)
    testCat = [testCatPath + file for file in testCat]
    testDog = listdir_nohidden(testDogPath)
    testDog = [testDogPath + file for file in testDog]
   
    train = trainCat + trainDog
    test = testCat + testDog

    trainImages = np.ndarray(shape=(len(train), ROW, COL, 3), dtype=np.float32)
    trainLabels = np.zeros(len(train))
    testImages = np.ndarray(shape=(len(test), ROW, COL, 3), dtype=np.float32)
    testLabels = np.zeros(len(test))
    
    # print('ready to start loading')
    
    i = 0
    for file in train:
        img = load_img(file, target_size=(ROW, COL))
        trainImages[i] = img_to_array(img)
        if 'dog' in file:
            trainLabels[i] = 1
        i = i + 1
        
    # print('done loading train')
    i = 0
    for file in test:
        img = load_img(file, target_size=(ROW, COL))
        testImages[i] = img_to_array(img)
        if 'dog' in file:
            testLabels[i] = 1
        i = i + 1
    # print('done loading test')
    
    valInd = random.sample(range(len(trainLabels)), math.ceil(len(trainLabels)/10))
    valImages = trainImages[valInd, :, :, :]
    valLabels = trainLabels[valInd]
    trainImages2 = trainImages[[i for i in range(len(trainImages)) if i not in valInd], :, :, :]
    trainLabels2 = trainLabels[[i for i in range(len(trainImages)) if i not in valInd]]

    np.save('trainImages', trainImages2)
    np.save('testImages', testImages)
    np.save('valImages', valImages)
    np.save('trainLabels', trainLabels2)
    np.save('testLabels', testLabels)
    np.save('valLabels', valLabels)

## Create and save numpy arrays for the CelebA dataset
##
## @description https://www.kaggle.com/jessicali9530/celeba-dataset
##
## @param pathToImages path to images containing original names from download
## @param pathToPartitions path to csv describing partitions
## @param ROW the desired number of rows for each image
## @param COL the desired number of columns for each image
##
def celebADataset(pathToImages, pathToPartitions, ROW, COL):
    split = pd.read_csv(pathToPartitions)
    
    trainFiles = np.array(split['image_id'])[np.array(split.loc[split['partition'] == 0].index)]
    valFiles = np.array(split['image_id'])[np.array(split.loc[split['partition'] == 1].index)]
    testFiles = np.array(split['image_id'])[np.array(split.loc[split['partition'] == 2].index)]        
    
    trainImages = np.ndarray(shape = (len(trainFiles), ROW, COL, 3), dtype=np.float32)
    valImages = np.ndarray(shape = (len(valFiles), ROW, COL, 3), dtype=np.float32)
    testImages = np.ndarray(shape = (len(testFiles), ROW, COL, 3), dtype=np.float32)
    
    i = 0
    for file in trainFiles:
        img = load_img(pathToImages + file, target_size=(ROW, COL))
        trainImages[i] = img_to_array(img)
        i = i + 1
    print("Done loading train")
    i = 0
    for file in valFiles:
        img = load_img(pathToImages + file, target_size=(ROW, COL))
        valImages[i] = img_to_array(img)
        i = i + 1
    print("Done loading validation")
    i = 0
    for file in testFiles:
        img = load_img(pathToImages + file, target_size=(ROW, COL))
        testImages[i] = img_to_array(img)
        i = i + 1
    print("Done loading test")
    
    
    attr = pd.read_csv('/home/bennettew/Documents/Winter2019/unkunk/NewDatasets/celebA/celeba-dataset/list_attr_celeba.csv')
    
    
    np.save('trainImages', trainImages)
    np.save('testImages', testImages)
    np.save('valImages', valImages)  
    
def zapposDatasetMaker(bootPath, sandalPath, shoePath, slipperPath, ROW, COL):

    ##
    ## Boots
    ##
    boot_images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(bootPath)) for f in fn]
    bootImages = np.ndarray(shape = (len(boot_images), ROW, COL, 3))
    bootLabels = np.zeros(len(boot_images), dtype=int)
    i = 0
    for file in boot_images:
        img = load_img(file, target_size = (ROW, COL))
        bootImages[i] = img_to_array(img)
        bootLabels[i] = 0
        i = i + 1

    print("Done with boots: " + str(len(bootLabels)))    

    ##
    ## Sandals
    ##
    sandal_images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(sandalPath)) for f in fn]
    sandalImages = np.ndarray(shape = (len(sandal_images), ROW, COL, 3))
    sandalLabels = np.zeros(len(sandal_images), dtype=int)
    i = 0
    for file in sandal_images:
        img = load_img(file, target_size = (ROW, COL))
        sandalImages[i] = img_to_array(img)
        sandalLabels[i] = 1
        i = i + 1

    print("Done with sandals: " + str(len(sandalLabels)))

    ##
    ## Shoes
    ##
    shoe_images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(shoePath)) for f in fn]
    shoeImages = np.ndarray(shape = (len(shoe_images), ROW, COL, 3))
    shoeLabels = np.zeros(len(shoe_images), dtype=int)
    i = 0
    for file in shoe_images:
        img = load_img(file, target_size = (ROW, COL))
        shoeImages[i] = img_to_array(img)
        shoeLabels[i] = 2
        i = i + 1

    print("Done with shoes: " + str(len(shoeLabels)))

    ##
    ## Slippers
    ##
    slipper_images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(slipperPath)) for f in fn]
    slipperImages = np.ndarray(shape = (len(slipper_images), ROW, COL, 3))
    slipperLabels = np.zeros(len(slipper_images), dtype=int)
    i = 0
    for file in slipper_images:
        img = load_img(file, target_size = (ROW, COL))
        slipperImages[i] = img_to_array(img)
        slipperLabels[i] = 3
        i = i + 1
    print("Done with slippers: " + str(len(slipperLabels)))
    
    dataset = np.concatenate((bootImages, shoeImages, sandalImages, slipperImages))
    labels = np.concatenate((bootLabels, shoeLabels, sandalLabels, slipperLabels))

    trainInd = random.sample(range(len(labels)), math.ceil(2*len(labels)/3))
    trainImages1 = dataset[trainInd, :, :, :]
    trainLabels1 = labels[trainInd]
    testImages = dataset[[i for i in range(len(labels)) if i not in trainInd], :, :, :]
    testLabels = labels[[i for i in range(len(labels)) if i not in trainInd]]
    valInd = random.sample(range(len(trainLabels1)), math.ceil(len(trainLabels1)/10))
    valImages = trainImages1[valInd, :, :, :]
    valLabels = trainLabels1[valInd]
    trainImages = trainImages1[[i for i in range(len(trainImages1)) if i not in valInd], :, :, :]
    trainLabels = trainLabels1[[i for i in range(len(trainImages1)) if i not in valInd]]
 
    np.save('trainImages', trainImages)
    np.save('testImages', testImages)
    np.save('valImages', valImages)
    np.save('trainLabels', trainLabels)
    np.save('testLabels', testLabels)
    np.save('valLabels', valLabels)

###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Model creators                              #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################


## CNN model maker
##
## @description This builds a CNN
##
## @param ROW input image rows
## @param COL input image columns
## @param CHANNEL input image channels
## @param CLASSES the number of classes
##
## @return a compiled Keras model
##
def modelMaker(ROW, COL, CHANNEL, CLASSES):
    model = Sequential()
    optimizer = Adam(lr=0.0001)
    #Convolution
    model.add(Conv2D(32, (3, 3),
                     input_shape = (ROW, COL, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'input',
                     kernel_regularizer=regularizers.l2()))
    model.add(Conv2D(32, (3, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_1',
                     kernel_regularizer=regularizers.l2()))
    #Pooling
    model.add(MaxPooling2D(pool_size = 2, data_format='channels_last', name = 'maxpool2d_1'))

    #ConvolutionLayer2
    model.add(Conv2D(64, (3, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_2',
                     kernel_regularizer=regularizers.l2()))
    model.add(Conv2D(64, (3, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_3',
                     kernel_regularizer=regularizers.l2()))

    model.add(MaxPooling2D(pool_size = 2, data_format='channels_last', name = 'maxpool2d_2'))

    model.add(Conv2D(128, (3, 3), 
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_4',
                     kernel_regularizer=regularizers.l2()))
    model.add(Conv2D(128, (3, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_5',
                     kernel_regularizer=regularizers.l2()))

    model.add(MaxPooling2D(pool_size = 2, data_format='channels_last', name = 'maxpool2d_3'))

    model.add(Conv2D(256, (3, 3), 
                     padding = 'same',
                     activation = 'relu', 
                     name = 'conv2d_6', kernel_regularizer=regularizers.l2()))
    model.add(Conv2D(256, (3, 3),
                     padding = 'same',
                     activation = 'relu',
                     name = 'conv2d_7',
                     kernel_regularizer=regularizers.l2()))

    model.add(MaxPooling2D(pool_size = 2, data_format='channels_last', name = 'maxpool2d_4'))

    #Flattening
    model.add(Flatten(name='flatten_1'))
                 
    model.add(Dense(units = 256, activation = 'relu', name = 'fc_1'))
    model.add(Dropout(rate=0.5, name='drop_1'))
    model.add(Dense(units = 256, activation = 'relu', name = 'fc_2'))
    model.add(Dropout(rate=0.5, name='drop_2'))
    model.add(Dense(units=CLASSES, name='output'))
        
    model.add(Activation(activation='softmax', name='activation_out'))

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return(model)

###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Model Calibration                           #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################

## Expected Confidence Error: https://arxiv.org/pdf/1706.04599.pdf
##
## @description This calculates the expected confidence error for a set of classifications
##
## @param confidence an array of confidences
## @param prediction an array of predictions
## @param trueLabel an array of the actual labels
## @param M the number of bins to use for the expected confidence error calculation
##
## @return the expected confidence error
##
def ECE(confidence, prediction, trueLabel, M):
    correct = prediction == trueLabel                                      ## create array of correct predictions
    m = np.linspace(1/M,1,M)                                               ## create bins
    minus = 1/M                                                                 
    part = np.zeros(1)                                                     ## location to hold results                          
    for i in m:
        BM = np.where( (confidence >= (i - minus)) & (confidence < i))[0]  ## indices of predictions in the bin
        if len(BM) > 0:
            acc = np.sum(correct[BM])/len(BM) * 100                        ## accuracy of predictions in the bin
            conf = np.sum(confidence[BM])/len(BM) * 100                    ## confidence of predictions in the bin
            coef = len(BM)/len(trueLabel)                                  ## the proportion of predictions in the bin
            part = np.append(part, coef * abs(conf-acc))                   
    return(np.nansum(part))

## Model to extract logits from a trained model
##
## @description Create a model to extract the logits from a trained model
##
## @param model trained model to extract logits from
## @param layerName the name of the logits layer
##
## @return A Keras model that will return the logits of the trained model
##     
def logitExtract(model, layerName):
     return(Model(inputs=model.input, outputs=model.get_layer(layerName).output))
    
    
## Learnable Keras divisor
##
## @description A custom Keras layer to divide inputs by a learnable parameter
##    
class LayerMDivide(Layer):
    def __init__(self, **kwargs):
        super(LayerMDivide, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.m = self.add_weight(name='m', 
                                 initializer=keras.initializers.Constant(value=1.5),
                                 shape = (),
                                 trainable=True)
        super(LayerMDivide, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x/self.m
    

    def compute_output_shape(self, input_shape):
        return (input_shape)
    
## Calibrate a model
##
## @description Calibrates a model with temperature scaling: : https://arxiv.org/pdf/1706.04599.pdf
##
## @param logitExtractor trained Keras model that returns the raw logits
## @param valImages numpy array of validation images
## @param valLabels numpy array of validation labels
## @param classes number of classes
## @param epochs number of epochs to learn the scaling parameter
##
## @return A calibrated Keras model
##        
def calibrateModel(logitExtractor, valImages, valLabels, classes, epochs):
    
    calibrate = Sequential()
    calibrate.add(Activation('linear', input_shape = (classes,))) ## Layer does nothing, but seems needed to initialize the model should try to fix this
    calibrate.add(LayerMDivide())                                 ## Custom layer define outside of this function which divides the logits by a learnable parameter
    calibrate.add(Activation('softmax'))                          ## Perform softmax on the scaled logits
    
    
    calibrate.compile(optimizer=keras.optimizers.RMSprop(), 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    calibrate.fit(logitExtractor.predict(valImages),
                  keras.utils.to_categorical(valLabels), 
                  batch_size = 128,
                  shuffle=True,
                  verbose=0,
                  epochs = epochs)
  
    return(calibrate)

###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Adversarial Distance                        #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################

## Calculate adversarial distance
##
## @description Calculate the distances between the number of gradient steps required to turn an image adversarial and the expected number of steps
def adversarialDistanceLOESS(gradientSteps, confidenceSteps):
    
    lowess = sm.nonparametric.lowess(gradientSteps, confidenceSteps, frac=.1)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # run scipy's interpolation
    f = interp1d(lowess_x, lowess_y, bounds_error=False)


    adversarialDistance = np.zeros(len(confidenceSteps))
    for point in range(len(confidenceSteps)):

        predicted = f(confidenceSteps[point]) 
        actual = gradientSteps[point]
        adversarialDistance[point] = actual - predicted
    return(adversarialDistance)


##
## Note you need to install randomgen==1.16
##
def boundaryMSE(classOfInterest, testImages, predictedTest, model, iterations):
    
    old_stdout = sys.stdout
    # Block print
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(old_stdout):
        sys.stdout = old_stdout
    
    
    catIndex = np.where(predictedTest == classOfInterest)[0]
    gradientSteps = np.zeros(len(catIndex))
    foolModel = foolbox.models.KerasModel(model, bounds=(0, 255))
    attack = foolbox.attacks.BoundaryAttack(model = foolModel, criterion = Misclassification(), distance=MAE)
    count = 0
    for i in catIndex:
        print("\r" + str(count) + " of " + str(len(catIndex)), end="")
        im = testImages[i,:,:,:].astype(np.float32)
        im = im[np.newaxis,...]
        blockPrint()
        adversarial_object = attack(im,
                                    np.array([classOfInterest]),
                                    unpack=False,
                                    verbose=False,
                                    iterations=iterations,
                                    log_every_n_steps = 200)
        enablePrint(old_stdout)
        distance = adversarial_object[0].distance
        distance = distance.value
        gradientSteps[count] = float(distance)
        count = count + 1       
    return(gradientSteps)


###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Searches                                    #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################
def randSearch(numToSearch, numToSelect):
        return(random.sample(range(numToSearch), numToSelect))
    
def lowConfidence(confidence, numToSelect):    
    return(confidence.argsort()[:numToSelect])

def highConfidence(confidence, numToSelect):    
    return(confidence.argsort()[-numToSelect:])

def lowAdversarialDistance(advDistance, numToSelect, confidence, highConfidence = 10000):
    
    ##
    ## Here we do a little extra work to allow us to remove high confidence points from the analysis
    ##
    ## Really we just want to sort the adversarial distance and return the index of the lowest values
    ##
    
    df = pd.DataFrame({'Index':range(len(confidence)),
                       'AdvDistance':advDistance,
                       'Confidence':confidence})
    df2 = df.sort_values('AdvDistance').loc[df['Confidence'] < highConfidence]
    return(df2['Index'].tolist()[:numToSelect])
    

def highAdversarialDistance(advDistance, numToSelect):
    return(advDistance.argsort()[-numToSelect:])


###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Utility Calcs                               #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################
def SDR(confidence, predictedLabels, trueLabels):
    numWrong = np.sum(trueLabels != predictedLabels)
    expected = len(trueLabels) - np.sum(confidence)
    if(expected == 0):
        print('what what what?')
        return(100000)
    return(numWrong/expected)


## Create a distance matrix
##
## @description matrix with distance between every element in X
##
## @param X set of features for points
##
## @return distance matrix with max distance being 1
##
def distMatrix(X):

    dists = euclidean_distances(X)
    dists = dists/np.max(dists)
    return(dists)

## Calculate spread
##
## @description calculate spread as described in paper 
##
## @param confidence numpy array of confidences
## @param predictedLabel numpy array of predictions
## @param trueLabel numpy array of actual labels
## @selected numpy array of queired instances
## @param distMatrix distance matrix with distances between all points
##
## @return facility utility
##
def spread(selected, distMatrix):
    
    ##
    ## Calculate spread
    ##
    spread = 0
    if np.sum(selected) == 0:
        spread = 1            ## Say spread is 1 if no unknown unknown
    else:
        minDist = np.min(distMatrix[np.ix_(range(distMatrix.shape[0]),selected)], axis = 1)   ## Find shortest distance from every point to queired instance
        spread = np.sum(minDist)/distMatrix.shape[0]   ## Sum shortest distances and average
    return(spread)        ## Return spread


###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Plots                                       #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################
## Create a reliability diagram
##
## @description Create a reliability diagram as described in https://arxiv.org/pdf/1706.04599.pdf
##
## @param confidence numpy array of confidences
## @param prediction numpy array of predictions
## @param trueLabel numpy array of actual labels
## @param nBins number of bins to create in the plot
##
## @return should print right to screen
##
def reliabilityDiagram(confidence, prediction, trueLabel, nBins):
    correct = prediction == trueLabel   ## Find correct predictions
    m = np.linspace(np.min(confidence), 1, nBins)      ## Split space into number of bins
    minus = m[1]-m[0]                   ## Find space between bins
    acc = np.zeros(nBins)               ## Place to store acc
    conf = np.zeros(nBins)              ## Place to store conf
    for i in range(len(m)-1):           ## Do this for every bin
        BM = np.where((confidence >= (m[i+1] - minus)) & 
                      (confidence < m[i+1]))[0] ## Find index of confidences 
        if len(BM) == 0:               ## Avoid NaNs
            acc[i] = 0
            conf[i] = 0
        else:
            acc[i] = np.sum(correct[BM])/len(BM)     ## Count correct predictions
            conf[i] = np.sum(confidence[BM])/len(BM) ## Sum confidences
    fig, ax = plt.subplots(1, 1)
    
    ax.set_ylim([0,1])
    ax.set_xlim([np.min(confidence),1])
    ax.bar(m, conf, color='red', width =minus, align = 'edge', edgecolor='black', linewidth = 2, label = 'Expected')
    ax.bar(m, acc, width =minus, align = 'edge', edgecolor='black', linewidth = 2, label = 'Actual')
    ax.grid()
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Accuracy')
    ax.legend(loc = 'upper left')
    ax.set_title('Reliability Diagram')

## Plot an image and it's adversary
##
## @description Create an adversarial image and plot it
##
## @param img an image to make adversarial
## @param model the model to use for an attack
## @note this will only work for an image of a dog with a cat dog classifier
def plotAdv(img, model):
    foolModel = foolbox.models.KerasModel(model, bounds=(0, 255))
    attack = foolbox.attacks.L1BasicIterativeAttack(foolModel, criterion=TargetClass(1), distance=MAE)
    adversarial_object = attack(img, label=0, unpack=False)
    advImg = adversarial_object.image

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(img/255)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(advImg/255)  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = ((advImg - img)/255)
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()


###############################################################################
###############################################################################
###############################################################################
#####                                                                   #######
#####                       Misc                                        #######
#####                                                                   #######
###############################################################################
###############################################################################
###############################################################################  
## Turn a numpy array of images black and white
##
## @description Create black and white images
##
## @param rgb a numpy array of images
##
## @return A numpy array where the first color channel is black and white and remainder are zeros
##         
def rgb2gray(rgb):
    dat1 = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    dat = np.empty((rgb.shape[0],rgb.shape[1], rgb.shape[2], 3))

    for k in range(len(dat)):
        dat[k,:,:,0] = dat1[k, :, :]
        dat[k,:,:,1] = np.zeros((dat1.shape[1], dat1.shape[2]))
        dat[k,:,:,2] = np.zeros((dat1.shape[1], dat1.shape[2]))
    return(dat)
     
      


    