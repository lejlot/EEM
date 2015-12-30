# EEM
Simple python implementation of **Extreme Entropy Machines**
http://link.springer.com/article/10.1007/s10044-015-0497-8#

## What is EEM?
Proposed model is a binary classifier belonging to the family of Randomized Neural Networks.
From technical perspective it is a 1-hidden layer neural network, which uses a gaussian
density estimator in the output layer based on Ledoit-Wolf covariance estimator to perform
information theoretic based optimization. 

## When to use EEM?
EEM is quite specific model, so make sure that it is well suited for your problem,
by answering following questions:

* It your problem a binary classification?
* Do you care about balanced accuracy (or GMean)?
* Do you need a fast, low-parametric model (possible at the cost of accuracy)?

If you answered **yes** for all the above - EEM is for you, have fun!

## What are the main strengths of the model?

* It is very simple to use.
* It learns rapidly.
* You get not only classification but also probability estimates.
* Wide range of activation functions can be used.
* It can be trained in an online-fashion efficiently (*not yet implemented*)

## Citing
```
@article{czarnecki2015eem,
    title={Extreme Entropy Machines: Robust information theoretic classification},
    author={Czarnecki, Wojciech Marian and Tabor, Jacek},
    journal={Pattern Analysis and Applications},
    year={2015},
    doi={10.1007/s10044-015-0497-8},
}
