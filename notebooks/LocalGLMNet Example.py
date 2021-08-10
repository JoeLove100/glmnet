# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: glmnet
#     language: python
#     name: glmnet
# ---

# %% [markdown]
# # LocalGLMNet demonstration

# %% [markdown]
# The aim of this notebook is to provide a brief demonstration of the uses of the LocalGLMNet model. While such deep learning techniques are better suited to larger and more complex tabular datasets, we use the classic Abalone dataset here for simplicty.

# %%
# add parent directory to path

import sys
sys.path.append("../")

# %%
# do imports

import numpy as np
import pandas as pd
from typing import Tuple
from local_glm_net import LocalGlmNet
from matplotlib import pyplot as plt

# %% [markdown]
# ### Use dummy data

# %%
np.eye(8)

# %%
rng = np.random.default_rng(1994)

def get_dummy_target(row: pd.Series) -> float:
    
    x_1 = row.iloc[0]
    x_2 = row.iloc[1]
    x_3 = row.iloc[2]
    x_4 = row.iloc[3]
    x_5 = row.iloc[4]
    x_6 = row.iloc[5]
    
    val =  0.5 * x_1 - 0.25 * (x_2 ** 2) + 0.5 * abs(x_3) * np.sin(2 * x_3) + \
           0.5 * x_4 * x_5 + 0.125 * (x_5 ** 2) * x_6 + rng.normal()
    return val
    
def get_dummy_data(size: int) -> Tuple[pd.DataFrame, pd.Series]:
    
    corr = np.eye(8)
    corr[1, 7] = corr[7, 1] = 0.5
    features = rng.multivariate_normal(mean=np.zeros(8), cov=corr, size=size)
    features = pd.DataFrame(features, columns=[f"feature_{i + 1}" for i in range(8)])
    target = features.apply(get_dummy_target, axis=1)
    return features, target

dummy_features, dummy_target = get_dummy_data(int(1e5))

# %%
glm_model = LocalGlmNet(shape=8, layer_shapes=[20, 15, 10], 
                        model_type="regression", layer_activation="tanh")
glm_model.fit(dummy_features, dummy_target, epochs=200, val_split=0.2, verbose=False, batch_size=8)

# %%
fig, axs = glm_model.plot_betas_by_feature(dummy_features.values, sample_size=0.05, 
                                           plot_random=True, plot_as_contributions=False)
fig.set_size_inches(20, 8)
fig.tight_layout()

# %%
fig, axs = glm_model.plot_interactions(features.values, sample_size=0.1, cols=2)
fig.set_size_inches(20, 8)

# %% [markdown]
# ### Read in and process Abalone data

# %% [markdown]
# We first need to read in the Abalone dataset from the TensorFlow datasets, and pre-process the features slighlty such that these are all normalised to have mean 0 and standard deviation 1.

# %%
# read in the raw data

raw_data = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv', 
                        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", 
                               "Shell weight", "Age"])

# %%
# check for missing data

raw_data.isnull().sum()

# %%
# check summary of data

raw_data.describe().T

# %%
# split into features/target and normalise the former

data = raw_data.copy(deep=True)
features, target = data.drop("Age", axis=1), data["Age"]

for col in list(features):
    mean = features[col].mean()
    std = features[col].std()
    features[col] = (features[col] - mean) / std


# %% [markdown]
# ### Create and fit our model

# %% [markdown]
# The next step is to create and fit our model. We go for a relatively simple model architecture with two hidden layers of size 16 and an output layer for our "attention" (beta) coefficients of size 7, in line with our features.  We then fit our model, ready to produce our GLM plots in the next sesion.  Note that the LocalGlmNet essenitally wraps a classic Keras neural network, and all training is handled by the standard Keras routines.

# %%
# define and fit model

glm_model = LocalGlmNet(shape=7, layer_shapes=[20, 15, 10, 40], 
                        model_type="regression", layer_activation="tanh")
glm_model.fit(features, target, epochs=200, verbose=False)

# %% [markdown]
# ### Create plots from model

# %% [markdown]
# Now we have a trained model we can use this to make a series of plots in order to better explore our data (and how our model uses it to make predictions). This is at the heart of "explainable" machine learning.  First of all, we can create a plot to show the distribution of the "attention" beta parameters for each feature at each point. Note the dotted lines, which show 95%, 99% and 99.9% central confidence intervals.

# %%
fig, axs = glm_model.plot_betas_by_feature(features.values, list(features), sample_size=1, plot_random=True)
fig.set_size_inches(15, 20)
plt.show();

# %% [markdown]
# In our second chart, we can plot the "interactions" between different features, where we represent these as the gradient of each "attention" beta with respect to each of the underlying variables.

# %%
features.shape

# %%
fig, axs = glm_model.plot_interactions(features.values, list(features), sample_size=1)
fig.set_size_inches(15, 20)
plt.show();

# %% [markdown]
# Finally, we can plot the feature importance, which we take to be the mean of the absolute values of the

# %%
fig, axs = glm_model.plot_feature_importance(features.values, list(features), sample_size=1)
fig.set_size_inches(8, 6)

# %%

# %%
