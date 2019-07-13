#!/usr/bin/env python
# coding: utf-8

# # Notebook Objectives
# - generate synthetic time-series [ using [TimeSynth](https://github.com/TimeSynth/TimeSynth) ]
# - visualize data
# - pre-process [ split train/test, and normalize ]
# - build model 
# - train model
# - evaluate model
# 

# In[149]:


import numpy as np
import matplotlib.pylab as plt


# # Import North Data

# In[150]:


dataset = np.load('./data/2018-01-01__2019-01-01__NConservatory_npWeekdayAllOrderedSensorsTimeRef.npy')


# In[188]:


dataset.shape


# In[153]:


#np.argwhere(dataset=-5.32848)


# In[169]:


date_index = np.load('./data/2018-01-01__2019-01-01__NConservatory_npWeekdayAllOrderedSensorsTimeRef_sampleBounds.npy')
### how we are gonna find dates of anomalies :D
date_index#.shape


# In[170]:


date_index.shape


# # Visualize Data at Multiple Scales

# In[154]:


def plot_data ( data ):    
    plt.figure(figsize = (10,15));     
    plt.subplot(4,1,1); plt.plot(data); plt.title('all data')
    plt.subplot(4,1,2); plt.plot(data[0:int(data.shape[0]//10.0)]); plt.title('first 10%')
    plt.subplot(4,1,3); plt.plot(data[0:int(data.shape[0]//100.0)]); plt.title('first 1%')
    plt.subplot(4,1,4); plt.plot(data[0:int(data.shape[0]//1000.0)], '-x'); plt.title('first .1%')    


# In[155]:


dataset.shape


# In[156]:


test = dataset[0]#[:400]


# In[157]:


test


# In[159]:


dataset


# In[158]:


#12:10-->
plot_data(test)


# # Data Prep -- split and normalize

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[25]:


def split_and_rescale ( X ):
    X_train, X_test = train_test_split( X, test_size = .25, shuffle = False )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform( X_train )
    X_test_scaled = scaler.transform( X_test )
    return X_train_scaled, X_test_scaled


# In[26]:


S1_train_scaled, S1_test_scaled = split_and_rescale(dataset)


# # Generate Sliding Windows and Shuffle

# In[27]:


def reshape_into_sliding_windows ( X, windowSize, advanceSamples = 1 ):
    # determine number of sliding windows that fit within dataset
    nWindows = int( np.floor( (X.shape[0] - windowSize)/(advanceSamples*1.0) ) )
    
    # pre-allocate matrix which holds sliding windows
    outputMatrix = np.zeros((nWindows, windowSize))
    
    # populate each sliding window
    for iWindow in range(nWindows):
        startIndex = iWindow * advanceSamples
        endIndex = startIndex + windowSize
        
        outputMatrix[iWindow, :] = X[ startIndex:endIndex, 0]
    
    return outputMatrix


# In[101]:


samplesPerSensorInput = 5000
S1_train_scaled_windowed = reshape_into_sliding_windows( S1_train_scaled, samplesPerSensorInput)
trainingData = S1_train_scaled_windowed


# In[102]:


trainingData.shape


# In[103]:


plt.plot(trainingData[0,:], '-') 


# # Model Building

# - be careful to move data to the GPU during training and back to the CPU for visualization
# - we need to make sure the data is in np.float32 format to matches the dtype of the model weights
# - don't forget to shuffle the data during training 

# In[104]:


import torch, torch.nn as nn, time
from torch.utils.data import Dataset, DataLoader


# # Define DataLoaders

# In[105]:


dataLoaderTrain = DataLoader( trainingData.astype('float32'), 
                                 batch_size = 16, 
                                 shuffle = True )

dataLoaderTest = DataLoader( trainingData.astype('float32'), 
                                 batch_size = 1, 
                                 shuffle = False )


# # Declare Model

# In[106]:


inputDimensionality = trainingData.shape[1]
inputDimensionality
#trainingData.shape


# In[107]:


inputDimensionality = trainingData.shape[1]

###MESS W THE BOTTLE NECKS :D
model = nn.Sequential (
    nn.Linear(inputDimensionality, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality//8), nn.Sigmoid(),
    nn.Linear(inputDimensionality//8, inputDimensionality//20), nn.Sigmoid(),
    nn.Linear(inputDimensionality//20, inputDimensionality//8), nn.Sigmoid(),
    nn.Linear(inputDimensionality//8, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality)
)


# In[108]:


#inputDimensionality = trainingData.shape[1]

###MESS W THE BOTTLE NECKS :D
#model = nn.Sequential (
#    nn.Linear(inputDimensionality, inputDimensionality//2), nn.Sigmoid(),
#    nn.Linear(inputDimensionality//2, inputDimensionality//4), nn.Sigmoid(),
#    nn.Linear(inputDimensionality//4, inputDimensionality//10), nn.Sigmoid(),
#    nn.Linear(inputDimensionality//10, inputDimensionality//4), nn.Sigmoid(),
#    nn.Linear(inputDimensionality//4, inputDimensionality//2), nn.Sigmoid(),
#    nn.Linear(inputDimensionality//2, inputDimensionality)
#)


# In[109]:


from autoencoder_tutorial.nnViz_pytorch import *


# In[110]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [9.5, 13]
plt.rcParams['figure.subplot.left'] = plt.rcParams['figure.subplot.bottom'] = .1
plt.rcParams['figure.subplot.right'] = plt.rcParams['figure.subplot.top'] = .9


# In[111]:


startTime = time.time()

plt.figure(figsize=(20,4)); ax = plt.gca()

visualize_model(model, ax)

plt.axis('tight'); plt.axis('off'); plt.show()

print('elapsed time: {}'.format(time.time()-startTime))


# # Determine Target Device for Training

# In[112]:


targetDeviceCPU = torch.device('cpu')
targetDeviceGPU = torch.device('cuda:0') 
targetDevice = targetDeviceGPU


# # Training Loop

# In[113]:


def train_model ( model, dataLoader, targeDevice, nEpochs = 10 ):

    model = model.to( targetDevice )
    
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters() )
    lossHistory = []
    
    # training loop    
    for iEpoch in range(nEpochs):   
        cumulativeLoss = 0
        for i, iInputBatch in enumerate( dataLoader ):
            
            # move batch data to target training device [ cpu or gpu ]
            iInputBatch = iInputBatch.to( targetDevice )
            
            # zero/reset the parameter gradient buffers to avoid accumulation [ usually accumulation is necessary for temporally unrolled networks ]
            optimizer.zero_grad()

            # generate predictions/reconstructions
            predictions = model.forward(iInputBatch)

            # compute error 
            loss = lossFunction( predictions, iInputBatch )
            cumulativeLoss += loss.item() # gets scaler value held in the loss tensor
            
            # compute gradients by propagating the error backward through the model/graph
            loss.backward()

            # apply gradients to update model parameters
            optimizer.step()
            
        print( 'epoch {} of {} -- avg batch loss: {}'.format(iEpoch, nEpochs, cumulativeLoss))
        
        lossHistory += [ cumulativeLoss ]
    return model, lossHistory


# ### Run Training Loop

# In[114]:


#12:27
startTime = time.time()

model, lossHistory = train_model( model, dataLoaderTrain, targetDevice, nEpochs = 100 )

print('elapsed time : {} '.format(time.time() - startTime))


# ### Visualize Progression of Learning

# In[161]:


plt.plot(lossHistory)
plt.title('Loss History'); plt.xlabel('epoch'); plt.ylabel('cumulative loss');


# # Evaluate Model Performance

# Simple demo using a single sample 

# In[116]:


sample = iter(dataLoaderTest).next()[0] # get first element from sample batch        
reconstruction = model.forward(sample.to(targetDevice))
plt.plot(sample.numpy())
plt.plot(reconstruction.data.cpu().numpy())


# We can also build an evaluation function that displays results and allows for multiple inferences

# In[117]:


def evaluate_model ( model, dataLoader, targetDevice, nEvals = 3):

    for iSample in range(nEvals):

        sample = iter(dataLoader).next()[0] # get first element from sample batch        
        reconstruction = model.forward(sample.to(targetDevice))

        sampleNumpy = sample.numpy()
        reconstructionNumpy = reconstruction.data.cpu().numpy()
        error = np.sqrt( (reconstructionNumpy - sampleNumpy)**2 ) #model.compile(loss='mean_squared_error')

        plt.figure(figsize=(9,5))
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)    

        ax1.plot(sampleNumpy)
        ax1.plot(reconstructionNumpy, '-.')
        ax1.set_title('sample {}, total error {}'.format(iSample, np.sum(error)))
        ax1.legend(['input data', 'reconstruction'])

        ax2.plot(error)
        ax2.legend( ['reconstruction error'] )


# In[118]:


evaluate_model( model, dataLoaderTest, targetDeviceGPU)


# In[163]:


input_data = sample.numpy()
#input_data


# In[177]:


input_data


# In[164]:


model_data = reconstruction.data.cpu().numpy()
#model_data


# In[178]:


model_data


# In[165]:


diffs = np.sqrt( (model_data - input_data)**2 ) #test = input_data - model_data ##works too
#diffs


# In[191]:


diffs.shape


# In[192]:


input_data.shape


# In[195]:


input_data[0]


# In[ ]:





# In[194]:


date_index.shape


# In[199]:


max_diff = 2
anomalies = np.where( diffs > max_diff  )  #locations of arrays that give anomalies
np.where( diffs > 1 )  #locations of arrays that give anomalies
diffs[ np.where( diffs > max_diff  ) ] #actual values > 2


# In[216]:


anomalies


# In[ ]:





# In[200]:


mylist = []
for i,val in enumerate(anomalies):
    #print(i)
    #print(val)
    mylist.append(date_index[val])
    


# In[201]:


mylist


# In[203]:


date_index


# In[210]:


import pandas as pd
result = pd.DataFrame(data=date_index[1:,1:],index=date_index[0:,0],columns=date_index[0,1:]) 


# In[212]:


result.reset_index()


# In[221]:


result['anomaly'] = 0


# In[222]:


result.head()


# In[224]:


mylist = []

for i,val in enumerate(anomalies):
    #print(i)
    #print(val)
    mylist.append(date_index[val])
    result['anomaly'].iloc[val] = 1
    


# In[225]:


result.iloc[324]


# In[226]:


#marks anomalies!
result.loc[result['anomaly'] == 1]


# In[232]:


result.info


# In[237]:


test = result[:20006]


# In[245]:


test.head()
test.loc[test['anomaly'] == 1]


# In[246]:


test.to_csv('south_conservatory_anomalies.csv',index=False)


# In[239]:


#!unzip testData.zip


# In[182]:


#testdata = np.load('test_data/nConservatory_2019_testData.npy')


# In[229]:


#testdata.shape


# In[230]:


#dataLoaderTest


# In[185]:


#testdataLoaderTest = DataLoader( testdata.astype('float32'), 
                                 batch_size = 1, 
                                 shuffle = False )


# In[228]:


#evaluate_model( model, testdataLoaderTest, targetDeviceGPU)

#pred = model.predict(testdata)
#xtest["prediciton"] = pred
#xtest.to_csv("my_new_file.csv")


# In[ ]:





# # TODO: Inject Synthetic Anomalies
# - morph 
# - replace
# - swap

# # Advanced Architecture Construction

# Note we can also build separate encoder and decoder modules, and combine them together into a final model. This is useful if we want to extract bottleneck activations
'''
# FYI this is also possible
inputDimensionality = trainingData.shape[1]
encoderModel = nn.Sequential(
    nn.Linear(inputDimensionality, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality//4), nn.Sigmoid(),
    nn.Linear(inputDimensionality//4, inputDimensionality//10), nn.Sigmoid()
)
decoderModel = nn.Sequential(
    nn.Linear(inputDimensionality//10, inputDimensionality//4), nn.Sigmoid(),
    nn.Linear(inputDimensionality//4, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality)
)

# combine
list_of_layers = list(encoderModel.children())
list_of_layers.extend(list(decoderModel.children()))
model = nn.Sequential (*list_of_layers)

# sanity check
list( nn.Sequential(*list(model.children())[0:6]).parameters() ) == list(encoderModel.parameters())
'''
# In[ ]:




