import numpy

#-----------Handle Command Line Input-------

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
    
locs = numpy.empty([file_len('testData.txt'),2], dtype=float)
classifications = numpy.empty([file_len('testData.txt'),1], dtype=float)

coordList = []
classificationList = []

# locs = (x, y), classifications = Score on test
for line in open('testData.txt','r'):
    splitLine = line[:-2].split(" ")
    x = float(splitLine[0])
    y = float(splitLine[1])
    coordList.append(numpy.array([x,y]))
    classificationList.append(float(splitLine[2]))
    
locs = numpy.array(coordList)
classifications = numpy.array(classificationList)

#-----------Neural Network Etc..------------

#New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):

        #Define nueral network size
        self.numInputNeurons = 2
        self.numHiddenNeurons = int(sys.argv[2])
        self.numOutputNeurons = 1
        
        #Weights (parameters)
        self.weightToHidden = numpy.random.randn(self.numInputNeurons,self.numHiddenNeurons)
        self.weightToOutput = numpy.random.randn(self.numHiddenNeurons,self.numOutputNeurons)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, locs):
        #Propogate inputs though network
        self.z2 = numpy.dot(locs, self.weightToHidden)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = numpy.dot(self.a2, self.weightToOutput)
        classificationHat = self.sigmoid(self.z3) 
        return classificationHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+numpy.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return numpy.exp(-z)/((1+numpy.exp(-z))**2)
    
    def costFunction(self, locs, classifications):
        #Compute cost for given locs,classifications, use weights already stored in class.
        self.classificationHat = self.forward(locs)
        J = 0.5*sum((classifications-self.classificationHat)**2)/locs.shape[0] + (self.Lambda/2)*(sum(self.weightToHidden**2)+sum(self.weightToOutput**2))
        return J
        
    def costFunctionPrime(self, locs, classifications):
        #Compute derivative with respect to W and weightToOutput for a given locs and classifications:
        self.classificationHat = self.forward(locs)
        
        delta3 = numpy.multiply(-(classifications-self.classificationHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdweightToOutput = numpy.dot(self.a2.T, delta3)/locs.shape[0] + self.Lambda*self.weightToOutput
        
        delta2 = numpy.dot(delta3, self.weightToOutput.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdweightToHidden = numpy.dot(locs.T, delta2)/locs.shape[0] + self.Lambda*self.weightToHidden
        
        return dJdweightToHidden, dJdweightToOutput
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get weightToHidden and weightToOutput Rolled into vector:
        params = numpy.concatenate((self.weightToHidden.ravel(), self.weightToOutput.ravel()))
        return params
    
    def setParams(self, params):
        #Set weightToHidden and weightToOutput using single parameter vector:
        weightToHidden_start = 0
        weightToHidden_end = self.numHiddenNeurons*self.numInputNeurons
        self.weightToHidden = numpy.reshape(params[weightToHidden_start:weightToHidden_end], \
                             (self.numInputNeurons, self.numHiddenNeurons))
        weightToOutput_end = weightToHidden_end + self.numHiddenNeurons*self.numOutputNeurons
        self.weightToOutput = numpy.reshape(params[weightToHidden_end:weightToOutput_end], \
                             (self.numHiddenNeurons, self.numOutputNeurons))
        
    def computeGradients(self, locs, classifications):
        dJdweightToHidden, dJdweightToOutput = self.costFunctionPrime(locs, classifications)
        return numpy.concatenate((dJdweightToHidden.ravel(), dJdweightToOutput.ravel()))

def computeNumericalGradient(N, locs, classifications):
        paramsInitial = N.getParams()
        numgrad = numpy.zeros(paramsInitial.shape)
        perturb = numpy.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(locs, classifications)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(locs, classifications)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.locs, self.classifications))   
        
    def costFunctionWrapper(self, params, locs, classifications):
        self.N.setParams(params)
        cost = self.N.costFunction(locs, classifications)
        grad = self.N.computeGradients(locs,classifications)
        return cost, grad
        
    def train(self, locs, classifications):
        #Make an internal variable for the callback function:
        self.locs = locs
        self.classifications = classifications

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(locs, classifications), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res