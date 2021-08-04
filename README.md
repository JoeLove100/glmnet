# glmnet

## Introduction
This is an implementation of the LocalGLMnet model suggested
by Ronald Richman and Mario W&uuml;thrich in their July 2021 paper
"*LocalGLMnet: interpretable deep learning for tabular data*". The idea behind 
this model is to create a framework for predictive analysis
on tabular data which combines the powerful modelling capabilities of 
feed-forward neuarl networks with the interpretability of GLMs.

## Using the package
The LocalGlmNet class essentially forms a wrapper around the basic keras
model, providing a convenient set of plotting functions to allow users to 
carry out feature importance and feature interaction analysis - the example
notebook in the repo provides more details on how this functionality can be used.
The .py file can be converted to a standard notebook using the 
[Jupytext package](https://github.com/mwouts/jupytext).
