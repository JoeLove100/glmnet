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
# # Example - French Motor Claims Data

# %% [markdown]
# Replicate the analysis in the initial paper.

# %%
# add parent dir to path

import sys
sys.path.append("../")

# %%
# do imports

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from local_glm_net import LocalGlmNet
from matplotlib import pyplot as plt

# %% [markdown]
# ## Get raw data and check

# %%
# request raw data from OpenML

# raw_data_frequency = pd.read_csv("https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff")
# raw_data_severity = pd.read_csv("https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff")

raw_data_frequency = pd.read_csv("frequency_local.csv", index_col=0)
raw_data_severity = pd.read_csv("severity_local.csv", index_col=0)

# %%
# check for missing data in frequency data

raw_data_frequency.isnull().sum()

# %%
# check for missing data in severity data

raw_data_severity.isnull().sum()

# %%
# show basic data summary for frequency data

raw_data_frequency.describe().T

# %%
# show similar summary for severity data

raw_data_severity.describe().T

# %% [markdown]
# ## Data cleaning

# %% [markdown]
# Apply cleaning and processing to the data as per section B1 of [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3822407) paper

# %%
data_frequency = raw_data_frequency.copy(deep=True)
data_severity = raw_data_severity.copy(deep=True)

# %%
data_frequency.head()

# %%
# ids for <= 24,500 don't all appear in severity so remove

data_frequency = data_frequency[data_frequency["IDpol"] > 24500]

# %%
# aggregate the claim severities

data_severity = data_severity.groupby("IDpol").sum().reset_index()

# %%
# merge the two data sets

data = pd.merge(data_frequency, data_severity, how="left", on="IDpol")
data["ClaimAmount"] = data["ClaimAmount"].fillna(0)

# %%
# remove drivers with claim counts > 5

data = data[data["ClaimNb"] <= 5]

# %%
# cap exposures at 1 year

data["Exposure"] = data["Exposure"].where(data["Exposure"] < 1, 1)

# %%
data["Area"].value_counts()

# %%
# map area codes to ordinals

data["Area"] = data["Area"].map({"'A'": 1, "'B'": 2, "'C'": 3, "'D'": 4, "'E'": 5, "'F'": 6}) 

# %%
# map diesel to binary

data["VehGas"] = data["VehGas"].map({"'Regular'": 0, "'Diesel'": 1})

# %%
# rename the columns to be easier to work with

data = data.rename(columns={"IDpol": "id", 
                            "ClaimNb": "claim_freq",
                            "Exposure": "exposure",
                            "Area": "area_code",
                            "VehPower": "vehicle_power",
                            "VehAge": "vehicle_age",
                            "DrivAge": "driver_age", 
                            "BonusMalus": "bonus_malus",
                            "VehBrand": "vehicle_brand",
                            "VehGas": "is_diesel",
                            "Density": "area_density",
                            "Region": "area_region",
                            "ClaimAmount": "claim_amount"})

# %%
cts_columns = ["area_code", "bonus_malus", "area_density", "driver_age", "vehicle_age", "vehicle_power"]
categorical_components = ["is_diesel", "vehicle_brand", "area_region"]
target = "claim_amount"

# %% [markdown]
# ## Preprocess the data

# %%
processed_data = data.copy(deep=True)

# %%
# standardise the continuous data

for col in cts_columns:
    mu = processed_data[col].mean()
    sigma = processed_data[col].std()
    processed_data[col] = (processed_data[col] - mu) / sigma


# %%
# one-hot encode the categorical variables

for col in categorical_components:
    if col == "is_diesel":
        continue  # already binary
    
    values = processed_data[col].unique()
    for val in values:
        processed_data[f"{col}_{val}"] = (processed_data[col] == val).astype("int")
    processed_data = processed_data.drop(col, axis=1)

# %%
# split into features and target (use claim freq for target for now)

features = processed_data.drop(["id", "claim_freq", "exposure", "claim_amount"], axis=1)
target = processed_data["claim_freq"]


# %% [markdown]
# ## Fit baseline NN model

# %% [markdown]
# We start by fitting a basic FFN to give a comparison for our LocalGLMNet model.

# %%
# create a baseline model to compare local GLM to

def get_baseline_model(input_shape: int,
                       model_name: str = "baseline_model"):
    
    input_layer = keras.Input(shape=(input_shape))
    x = keras.layers.Dense(20, activation="tanh", name="hidden_1")(input_layer)
    x = keras.layers.Dense(15, activation="tanh", name="hidden_2")(x)
    x = keras.layers.Dense(10, activation="tanh", name="hidden_3")(x)
    output_layer = keras.layers.Dense(1, activation="relu", name="output_layer")(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name=model_name)
    model.compile(optimizer="adam", loss="poisson")
    return model

baseline_model = get_baseline_model(input_shape=features.shape[1])

# %%
# fit and evaluate the baseline model

early_stop = keras.callbacks.EarlyStopping(patience=2)

MAX_EPOCHS = 100
# baseline_model.fit(x=features, y=target, epochs=MAX_EPOCHS, validation_split=0.1, callbacks=[early_stop])

# %% [markdown]
# ## Fit LocalGLMNet model

# %%
random_gen = np.random.default_rng(1234)
local_glm = LocalGlmNet(shape=features.shape[1], layer_shapes = [20, 15, 20], model_type="poisson", 
                        random_generator=random_gen, layer_activation="tanh")

# %%
local_glm.fit(features, target, 0.1, MAX_EPOCHS, use_early_stop=True, batch_size=5000)

# %% [markdown]
# ## Make diagnostic plots

# %%
fig, ax = local_glm.plot_betas_by_feature(x_data=features.values, features_to_plot=cts_columns, cols=2, 
                                          sample_size=0.2, plot_random=True)
fig.set_size_inches(15, 20)
plt.show()

# %%
fig, ax = local_glm.plot_betas_by_feature(x_data=features.values, features_to_plot=cts_columns, cols=2, 
                                          sample_size=0.2, plot_random=True, plot_as_contributions=True,
                                         y_lim=(-5, 3))
fig.set_size_inches(15, 20)
plt.show()

# %%
fig, ax = local_glm.plot_interactions(x_data=features.values, features_to_plot=cts_columns, sample_size=0.25)
fig.set_size_inches(15, 20)
plt.show()

# %%
fig, ax = local_glm.plot_feature_importance(features.values, cts_columns)
fig.set_size_inches(10, 10)
plt.show()

# %%
fig, axs = local_glm.plot_categorical_betas(features.values, features_to_plot=categorical_components, cols=2)
fig.set_size_inches(20, 15)

# %%
