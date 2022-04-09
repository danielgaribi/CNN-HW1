r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
    Yes, increasing k lead to improved generalization for unseen data. If K is small,
    for example 1 on 2, we get an overfit model. The modle findes to closest train item
    to the givven input, if we have noise in our training set it affects more when K is
    small.
    When we set K to high the model became irrelevent, for example if we take k to be a
    the size of training set, every digit we will input to the model will be comapred
    to all the training items, and we will output the most common label in our training
    set for any input.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**
1.
"""

part3_q3 = r"""
**Your answer:**
1. The learning rate is good, the loss decends fast on the first epoches and then stay in a stady state.
if it was too low i expect the loss to decend slowly and to not reach it's mean value be the end of the trainig (need more epoches).
if iw was too high i expecd the loss to decends fast on the epoches, but then to change it value on every epoch (will not arrive to a stady state).

2. The model is Slightly overfitted to the training set becaues the traing and the training accurecy is higher then the test accurecy by 5 precent.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
