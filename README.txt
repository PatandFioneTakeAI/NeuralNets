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



