AUTHORS:  Fiona Heaney & Pat Lebold
ASSIGMENT: 2 - Artifical Neural Networks
***** Granted 2 day extension from original due date by Prof Heffernan (2/2) ********

TO RUN: python ann.py <filename> [h<number of hidden nodes> | p <holdout percentage> ]
where <filename> would specify the file to read from, h <number of hidden nodes> specifies the number of nodes used in the hidden layer and p <holdout percentage> is the percentage of the data that should be withheld to use for testing purposes.

QUESTIONS: 

1) Hidden neurons set to 5, with different hold out percentages. (We took 3 answers and averaged the error rates)
 h = 5 p = .1 --> .511, .461, .438  AVG = .470
 h = 5 p = .3 --> .507, .429, .5    AVG = .479
 h = 5 p = .5 --> .621, .51, .5     AVG = .543
 h = 5 p = .7 --> .516, .616, .5    AVG = .544
 h = 5 p = .9 --> .610, .445, .75   AVG = .6

Decreasing the hold out percentage decreased our error rate. This makes sense becaus ehaving more data to train on should make for a better ANN

2)Hidden neurons ranging from 2-10, with witholding = .2 

h = 2 	0.500,	0.500,	0.480  AVG = 	0.493
h = 3 	0.463,	0.500,	0.507  AVG = 	0.490
h = 4	 0.506,	0.481,	0.444  AVG = 	0.477
h = 5	 0.525,	0.431,	0.500  AVG = 	0.485
h = 6	 0.431,	0.406,	0.500  AVG = 	0.446
h = 7	 0.438,	0.506,	0.494  AVG =  0.479
h = 8	 0.575,	0.413,	0.506  AVG = 	0.498
h = 9	 0.419, 0.367,	0.481  AVG = 	0.423
h = 10	0.394,	0.394,	0.3462 AVG =  0.378


We think it’s safe to say our data didn’t produce data as smoothly as we would have wanted but ideally there would have been a decrease in error as number of hidden nodes increased. We would expect this because an increase in the number of hidden nodes allows us to better train each weight for the nodes. 

3) When implementing our neural network we decided that we would only do 200 iterations, given that the data set provided was only 200 entries long. We currently have the number of input and output nodes hard coded, but our model should be able to handle changes because we implemented matrices for propagation that can be dynamically changed but our accuracy would probably suffer as a result. 





