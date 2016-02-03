# Authors: Patrick Lebold & Fiona Heaney
# Project: Neural Network

import numpy as np
import sys

class NeuralNetwork(object):
    
    #Initialize random weights and network structure
    def __init__(self,numHiddenNeurons):        
        self.numHiddenNeurons = numHiddenNeurons
        self.inputWeights = np.random.randn(2,self.numHiddenNeurons)
        self.hiddenWeights = np.random.randn(self.numHiddenNeurons,1)
        
    #Push data through network
    def propogate(self, inputs):
        self.compoundedInputs = np.dot(inputs, self.inputWeights)
        self.activatedData = self.activationFunction(self.compoundedInputs)
        self.compoundedData = np.dot(self.activatedData, self.hiddenWeights)
        finalResult = self.activationFunction(self.compoundedData) 
        return finalResult
        
    #Sigmoid Activation Function
    def activationFunction(self, data):
        return 1/(1+np.exp(-data))
    
    #Derivative of Function (required for back-prop)
    def activationFunctionDerivative(self,data):
        return np.exp(-data)/((1+np.exp(-data))**2)
        
    #choo choo
    def train(self, inputTests, yTests):
        for iteration in range(0,200):
            #run test data through network
            self.finalResult = self.propogate(inputs)
            
            #perform backwards propogation
            compoundedErrorFromHidden = np.multiply(-(outputs-self.finalResult), self.activationFunctionDerivative(self.compoundedData))
            compoundedErrorFromInput = np.dot(compoundedErrorFromHidden, self.hiddenWeights.T)*self.activationFunctionDerivative(self.compoundedInputs)
            hiddenWeightsError = np.dot(self.activatedData.T, compoundedErrorFromHidden)
            inputWeightsError = np.dot(inputs.T, compoundedErrorFromInput)
            
            #Update weights based on error values
            np.subtract(self.inputWeights,inputWeightsError/2)
            np.subtract(self.hiddenWeights,hiddenWeightsError/2)
        
#--------------------------------------------------------------------#
# Handle code stuff

argsize = len(sys.argv)
filename = sys.argv[1]
h = 5
p = .2

#handle arguments
if argsize == 4:
    if sys.argv[2] == "h":
        h = int(sys.argv[3])
    else:
        p = 1-float(sys.argv[3])
elif argsize ==  6:
    if sys.argv[2] == "h":
        h = int(sys.argv[3])
        p = 1-float(sys.argv[5])
    else:
        h = int(sys.argv[5])
        p = 1-float(sys.argv[3])


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

filelen = file_len(filename)

coordList = []
classificationList = []

coordListResults = []
classificationListResults = []

linenumber = 0
#read file and make an arrays for data and classifications
for line in open(filename,'r'):
    splitLine = line[:-2].split(" ")
    x = float(splitLine[0])
    outputs = float(splitLine[1])
    #keep some data witheld 
    #testing data
    if linenumber >= filelen * p:
        coordListResults.append(np.array([x,outputs]))
        classificationListResults.append([float(splitLine[2])])
   #training data
    else:
        coordList.append(np.array([x,outputs]))
        classificationList.append([float(splitLine[2])])
    linenumber = linenumber + 1 

    
inputs = np.array(coordList)
outputs = np.array(classificationList)
inputResults = np.array(coordListResults)
yResults = np.array(classificationListResults)

# Normalize 
inputs = (inputs-np.amin(inputs, axis=0))/np.amax(inputs, axis=0)

# Let's make a neural network!
network = NeuralNetwork(h)
network.train(inputs,outputs)

#round normalized data to 1 or 0
results = np.rint(network.propogate(inputResults))

numResults = len(results)
numCorrect = 0

#count num correct for percentage error
for n in range(0,numResults-1):
    if(yResults[n] == results[n]):
        numCorrect = numCorrect + 1
        
percentWrong = 1-float(numCorrect)/float(numResults)
print("Error: "+str(percentWrong)+"%")
