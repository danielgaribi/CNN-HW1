r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
    Yes, increasing k lead to improved generalization for unseen data, when K values are are small. For example 1 on 2, we get an overfit model. The modle findes to closest train item to the given input, if we have noise in our training set it affects more when K is small.
    
    
When we set K to high the model became irrelevent, for example if we take k to be a
the size of training set, every digit we will input to the model will be comapred
to all the training items, and we will output the most common label in our training
set for any input.


As we can see in the graph above, after the k=3 incressing the k doesn't lead to generalization, it leads to overfit.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
Delta can be seleced arbitrary because the parameter labmda has the same purpose.

The purpose of delta is to mark the margin boundary of each seperator. Samples that cross the margin are penalized and increase our loss function. In a simillar manner, the lambda parameter also controls the penalty magnitude of large weights(seperators). We can deduct that changing delta (the margin from the seperator) to be bigger means increasing our seperator weights in order to decreas the panelty caused by delta. Increasing the wights also increase thus the lambda penalty increases.

symmetrically decreasing the delta has an opposite affect. So the tradeoff between the lambda penalty and delta allows delta to be arbitary
"""

part3_q2 = r"""
**Your answer:**
1. We can easily see that the yellow areas in the images is the number that the class represent. The darker areas represent areas that the number almost never will be at.
The images that were label inccorectly has something in common, the white number is overlapping other class yellow area better then the correct class.

2. This interpretaion is different from KNN in the comperration to the training set. KNN compares all the pixels of the input the all the pixel of each image in the training set, and finds the nearest neighbors. This interpretaion compare the each pixel of the input image to the same pixel in all the training set, and labeled it accordingly.
Another differce is that KNN need all the data set in order to label the input, while linear classofier use represention of the dataset.
"""

part3_q3 = r"""
**Your answer:**
1. The learning rate is good, the loss decends fast on the first epoches and then stay in a steady state.
if it was too low i expect the loss to decend slowly and to not reach it's mean value be the end of the trainig (need more epoches).
if it was too high i expecd the loss to decends fast on the epoches, but then to change it value on every epoch (will not arrive to a steady state).

2. The model is Slightly overfitted to the training set becaues the traing and the training accurecy is higher then the test accurecy by about 3-4 precent.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
Idealy we would like to see all the point on the y-y^ axis ()

The ideal pattern to see in a residual plot is just horizontal line, which indicates that all losses are zero, which means explained the data perfectly.

In our plot, the residual are centerned in a small margin from the y-y^ axis, as required. It's fairly a good residual plot.

Comapring to the top-5 features, our none linaer model seems to explaing the data beter, it has a beter rsq value (closer to one), while the top 5 features has rsq value about 0.5 or even less for the 3-5 plots, which means our none-liear model is better.
"""

part4_q2 = r"""
**Your answer:**
1. We think that you used logspace instead of linspace for lambda because for small value of lambdas, even small difference in labmda can change the loss dramatically. On the ather hand, when the value of lambda is big, small difference in labmda doesn't affects the loss.
So, using logspace allows are to try more small labmda values, while not wasting training time on many big lambdas.

2. We have 3 different degree parameter, 20 lambdas and 3 folds.
So in total, we fit to data 20 * 3 * 3 = **180 times**
"""

# ==============
