import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Dict, Optional
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

_VALID_ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear"]
_VALID_MODEL_TYPES = {"regression": "linear",
                      "binary_classification": "sigmoid",
                      "poisson": "exponential"}


class LocalGlmNet:

    # region private methods
    def _create_keras_model(self) -> Tuple[keras.models.Model,
                                           keras.models.Model]:

        inputs = keras.Input(shape=self._shape, name="input")

        x = None
        for i, m in enumerate(self._layer_shapes):
            layer_name = f"hidden_{i}"
            if x is None:
                x = keras.layers.Dense(m, activation=self._layer_activation,
                                       name=layer_name)(inputs)
            else:
                x = keras.layers.Dense(m, activation=self._layer_activation,
                                       name=layer_name)(x)

        x = keras.layers.Dense(self._shape, activation="linear",
                               name="betas")(x)
        beta_model = keras.models.Model(inputs=inputs, outputs=x)
        linear = keras.layers.Dot(axes=[1, 1],
                                  name="linear")([x, inputs])
        final_activation = _VALID_MODEL_TYPES[self._model_type]
        outputs = keras.layers.Activation(final_activation,
                                          name="link")(linear)
        model = keras.models.Model(inputs=inputs, outputs=outputs,
                                   name="LocalGLMNet")

        # TODO: should allow parameters in the below to be
        #   customised by user rather than set automatically
        if self._model_type == "regression":
            model.compile(optimizer="adam", loss="mean_squared_error")
        elif self._model_type == "binary_classification":
            model.compile(optimizer="adam", loss="binary_crossentropy")
        elif self._model_type == "poisson":
            model.compile(optimizer="adam", loss="poisson")

        return model, beta_model

    def _get_confidence_intervals(self) -> Dict[str, Tuple[float, float, str]]:

        quantiles = [("95%", 1.960, "yellow"),
                     ("99%", 2.807, "orange"),
                     ("99.9%",  3.291, "red")]
        out = dict()
        for name, q, color in quantiles:
            out[name] = (-q * self.conf_, q * self.conf_, color)
        return out

    def get_sampled_data(self,
                         x_data: np.ndarray,
                         sample_size: float) -> np.ndarray:

        number_of_rows = int(x_data.shape[0] * sample_size)
        sample_rows = np.sort(self._rng.choice(x_data.shape[0],
                                               size=number_of_rows,
                                               replace=False))
        rand_col = self._rng.normal(size=(number_of_rows, 1))
        x_data_sample = x_data[sample_rows, :]
        x_data_sample = np.concatenate([x_data_sample, rand_col], axis=1)
        return x_data_sample

    def check_plot_arguments(self,
                             feature_names: List[str],
                             sample: float,
                             is_categorical: bool = False) -> None:

        # check sample proportion is valid
        if sample <= 0 or sample > 1:
            raise ValueError(f"Sample should be in interval (0, 1], "
                             f"but is {sample}")

        # check we actually have data for all features to plot
        if is_categorical:
            features_not_recognized = []
            for ft in feature_names:
                encoded_features = [col_name for col_name in self.col_indices_
                                    if col_name.startswith(ft)]
                if not encoded_features:
                    features_not_recognized.append(ft)
        else:
            features_not_recognized = [ft for ft in feature_names if
                                       ft not in self.col_indices_]

        if features_not_recognized:
            raise ValueError(f"The following features were not recognized: "
                             f"{', '.join(features_not_recognized)}")

    # endregion

    def __init__(self,
                 shape: int,
                 layer_shapes: List[int],
                 model_type: str,
                 random_generator: Optional[np.random.Generator] = None,
                 layer_activation: str = "relu") -> None:
        """
        wrapper class for the LocalGLMNet architecture with
        attached utility methods for feature selection/importance
        and visualising feature interactions

        :param shape: number of features in the data (ie the
        shape of input expected by the wrapped neural net)
        :param layer_shapes: the number of units in each of the
        hidden layers, the last of which should be of equal dimension
        to the model shape
        :param model_type: the form of the output of the model (ie
        classification, regression or poisson count)
        :param random_generator: numpy random number generator
        :param layer_activation: activation function for the hidden
        layers
        """

        if not layer_shapes:
            raise ValueError("Must provide shape for at least "
                             "one hidden layer")

        if layer_activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"Activation function {layer_activation} "
                             f"is not supported: valid choices "
                             f"are {', '.join(_VALID_ACTIVATIONS)}")

        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"Model type {model_type} is not "
                             f"supported: valid choices "
                             f"are {', '.join(_VALID_MODEL_TYPES)}")

        if random_generator is None:
            random_generator = np.random.default_rng()

        # model parameters (adjust for random col)
        self._shape = shape + 1  # adjust for random col
        self._layer_shapes = layer_shapes
        self._model_type = model_type
        self._layer_activation = layer_activation
        self._rng = random_generator

        # only set after fitting
        self.prediction_model_ = None
        self.beta_model_ = None
        self.conf_ = None
        self.col_indices_ = {"Random": shape}

    def fit(self,
            x_train: pd.DataFrame,
            y_train: np.ndarray,
            val_split: float = 0.1,
            epochs: int = 100,
            use_early_stop: bool = True,
            **kwargs) -> None:
        """
        fit the wrapped neural network model using the
        standard Keras training loop

        :param x_train: numpy array of features to be used in fitting
        model weights
        :param y_train: numpy array of response values to be used in
        fitting model weights
        :param val_split: % of the training data that should be held back
        for validation purposes when training model weights
        :param epochs: number of training epochs
        :param use_early_stop: boolean flag for whether or not to add an
        early stopping callback
        """

        self.col_indices_.update({col: i for i, col in
                                  enumerate(x_train.columns)})
        x_train = x_train.values
        self.prediction_model_, self.beta_model_ = self._create_keras_model()
        random_col = self._rng.normal(size=(x_train.shape[0], 1))
        x_train = np.concatenate([x_train, random_col], axis=1)

        callbacks = []
        if use_early_stop:
            early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                       patience=2)
            callbacks.append(early_stop)

        self.prediction_model_.fit(x=x_train, y=y_train, epochs=epochs,
                                   validation_split=val_split,
                                   callbacks=callbacks, **kwargs)

        betas = self.beta_model_(x_train)
        self.conf_ = np.std(betas[:, -1])

    def plot_betas_by_feature(self,
                              x_data: np.ndarray,
                              features_to_plot: Optional[List[str]] = None,
                              sample_size: float = 0.25,
                              cols: int = 3,
                              plot_random: bool = False,
                              plot_as_contributions: bool = False,
                              y_lim: Tuple[float, float] = (-1, 1)) \
            -> [plt.Figure, plt.Axes]:
        """
        utility function to plot the distribution of the local
        beta features given a set of feature values. Can optionally plot
        confidence intervals to help with feature selection,

        :param x_data: data for which we would like to plot our
        beta values
        :param features_to_plot: names for the features in our x_data
        that we would like to plot
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        :param cols: number of columns in our plot
        :param plot_random: boolean flag for whether or not we would like
        to plot a figure for the additional random variable
        :param plot_as_contributions: boolean flag which, if true, plots
        contributions rather than outright beta values
        :param y_lim: y axis min and max values to scale plots properly
        """

        # get model betas
        if not features_to_plot and plot_random:
            features_to_plot = list(self.col_indices_)
        elif not features_to_plot:
            features_to_plot += [ft for ft in self.col_indices_ if
                                 ft != "Random"]
        elif plot_random:
            features_to_plot = features_to_plot + ["Random"]

        self.check_plot_arguments(features_to_plot, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        betas = self.beta_model_(x_data_sample)
        col_indices = [self.col_indices_[ft] for ft in features_to_plot]
        betas = betas.numpy()[:, col_indices]
        x_data_sample = x_data_sample[:, col_indices]

        if plot_as_contributions:
            betas = x_data_sample * betas

        # set up our grid for plotting
        rows = betas.shape[1] // cols
        if betas.shape[1] % cols != 0:
            rows += 1
        fig, axs = plt.subplots(rows, cols)

        # set up our confidence intervals
        conf_intervals = self._get_confidence_intervals()

        # plot scatter chart for each variable
        for i, feature_name in enumerate(features_to_plot):

            r, c = i // cols, i % cols
            ax = axs[r, c]
            ax.scatter(x_data_sample[:, i], betas[:, i])
            if plot_as_contributions:
                ax.set_title(f"Contributions for {feature_name}")
            else:
                ax.set_title(f"Coefficients for {feature_name}")
            ax.set_ylim(*y_lim)

            if not plot_as_contributions:
                for name, definition in conf_intervals.items():
                    lower, upper, color = definition
                    ax.axhline(lower, color=color, linestyle="--")
                    ax.axhline(upper, color=color, linestyle="--")

        return fig, axs

    def plot_interactions(self,
                          x_data: np.ndarray,
                          features_to_plot: Optional[List[str]] = None,
                          sample_size: float = 1,
                          cols: int = 2) -> Tuple[plt.Figure, plt.Axes]:
        """
        for each feature we plot a graph showing how the gradients of
        the betas vary with the value of the given feature

        :param x_data: data for which we would like to plot our
        beta values
        :param features_to_plot: names for the features in our x_data
        that we would like to plot
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        :param cols: number of columns in our plot
        """

        # calculate gradients based on sample
        if features_to_plot is None:
            features_to_plot = list(self.col_indices_)
        self.check_plot_arguments(features_to_plot, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        input_tensor = tf.convert_to_tensor(x_data_sample)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            beta = self.beta_model_(input_tensor)
        grads = tape.batch_jacobian(beta, input_tensor)
        col_indices = [self.col_indices_[ft] for ft in features_to_plot]
        grads_np = grads.numpy()[np.ix_(range(grads.shape[0]),  col_indices,
                                        col_indices)]
        x_data_sample = x_data_sample[:, col_indices]

        # set up our axes for plotting
        rows = int(math.ceil(len(features_to_plot) / cols))
        fig, axs = plt.subplots(rows, cols)

        # outer loop: plot interactions for each feature
        for i, feature_name in enumerate(features_to_plot):

            # get axes for plotting
            row, col = i // cols, i % cols
            ax = axs[row, col]
            ax.set_ylim(-0.7, 0.7)  # TODO: change limits based on data
            ax.set_title(f"Interactions for feature {feature_name}",
                         fontsize=15)

            # select feature values and gradients, and reorder
            d_beta_0 = grads_np[:, i, :]
            x_0 = x_data_sample[:, i]
            order = np.argsort(x_0)
            x_0 = x_0[order]
            d_beta_0 = d_beta_0[order, :]
            x_vals = np.linspace(np.min(x_0), np.max(x_0), 30)

            # inner loop: plot each beta
            for j, other_feature_name in enumerate(features_to_plot):
                spline = UnivariateSpline(x_0, d_beta_0[:, j])
                ax.plot(x_vals, [spline(v) for v in x_vals],
                        label=other_feature_name)

            ax.legend(loc="lower right", ncol=2)

        return fig, axs

    def plot_feature_importance(self,
                                x_data: np.ndarray,
                                features_to_plot: Optional[List[str]] = None,
                                sample_size: float = 1) \
            -> Tuple[plt.Figure, plt.Axes]:
        """
        for each feature we plot a graph showing how the gradients of
        the betas vary with the value of the given feature

        :param x_data: data for which we would like to plot our
        beta values
        :param features_to_plot: names for the features in our x_data
        that we would like to plot
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        """

        # get betas and calculate feature importance
        if features_to_plot is None:
            features_to_plot = list(self.col_indices_)
            features_to_plot.remove("Random")  # only plot random threshold

        self.check_plot_arguments(features_to_plot, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        col_indices = [self.col_indices_[ft] for ft in features_to_plot] + [-1]
        betas = self.beta_model_(x_data_sample).numpy()[:, col_indices]
        avg_abs_betas = abs(betas[:-1]).mean(axis=0)
        importance, threshold = avg_abs_betas[:-1], avg_abs_betas[-1]

        # reorder data and feature names for plotting
        order = np.argsort(importance)
        importance = importance[order]
        features_to_plot = [features_to_plot[i] for i in order]

        # set up axes and make bar plot
        fig, ax = plt.subplots()
        positions = np.arange(len(features_to_plot))
        ax.barh(positions, importance)
        ax.set_yticks(positions)
        ax.set_yticklabels(features_to_plot)
        ax.axvline(threshold, color="red", linestyle="--")

        return fig, ax

    def plot_categorical_betas(self,
                               x_data: np.ndarray,
                               features_to_plot: Optional[List[str]] = None,
                               sample_size: float = 1,
                               cols: int = 2) \
            -> Tuple[plt.Figure, plt.Axes]:
        """
        plot box and whisker plots for each of the categorical
        variables. For this to work, categorical features must be
        one-hot encoded in the form {feature_name}_{value}

        :param x_data: data for which we would like to plot our
        beta values
        :param features_to_plot: names for the features in our x_data
        that we would like to plot
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        :param cols: number of columns in resulting plot
        """

        # set features if not defined and check inputs
        if features_to_plot is None:
            features_to_plot = list(self.col_indices_)
            features_to_plot.remove("Random")  # only plot random threshold
        self.check_plot_arguments(features_to_plot, sample_size,
                                  is_categorical=True)

        # get data sample and use it to calculate betas
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        betas = self.beta_model_(x_data_sample).numpy()

        # set up axes
        rows = int(math.ceil(len(features_to_plot) / cols))
        fig, axs = plt.subplots(rows, cols)

        # plot box and whisker for each feature
        for i, feature in enumerate(features_to_plot):

            r, c = i // cols, i % cols
            ax = axs[r, c]

            categories = [col_name for col_name in list(self.col_indices_)
                          if col_name.startswith(feature)]
            category_names = [c.split("_")[-1] for c in categories]
            category_indices = [self.col_indices_[c] for c in categories]
            betas_for_feature = betas[:, category_indices]
            ax.boxplot(betas_for_feature, labels=category_names)

        return fig, axs
