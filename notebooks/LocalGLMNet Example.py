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
# do imports

import pandas as pd
from local_glm_net import LocalGlmNet
from matplotlib import pyplot as plt

# %% [markdown]
# ### Read in and process data

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
fig, axs = glm_model.plot_betas_by_feature(features.values, list(features), sample_size=1)
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
