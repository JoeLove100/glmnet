import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Dict, Optional
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

_VALID_ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear"]


class LocalGlmNet:

    # region private methods
    def _create_keras_model(self) -> Tuple[keras.models.Model,
                                           keras.models.Model]:

        inputs = keras.Input(shape=self._shape, name="input")

        x = None
        for i, m in enumerate(self._layer_shapes):
            layer_name = f"hidden_{i}"
            if x is None:
                x = keras.layers.Dense(m, activation=self._layer_activation, name=layer_name)(inputs)
            else:
                x = keras.layers.Dense(m, activation=self._layer_activation, name=layer_name)(x)

        beta_model = keras.models.Model(inputs=inputs, outputs=x)
        linear = keras.layers.Dot(axes=(1,), name="linear")([x, inputs])
        final_activation = "sigmoid" if self._is_classifier else "linear"
        outputs = keras.layers.Activation(final_activation, name="link")(linear)
        model = keras.models.Model(inputs=inputs, outputs=outputs,
                                   name="LocalGLMNet")

        # TODO: should allow parameters in the below to be
        #   customised by user rather than set automatically
        if self._is_classifier:
            model.compile(optimizer="adam", loss="binary_crossentropy",
                          metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="mean_squared_error")

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
        sample_rows = np.sort(self._rng.choice(x_data.shape[0], size=number_of_rows, replace=False))
        rand_col = self._rng.normal(size=(number_of_rows, 1))
        x_data_sample = x_data[sample_rows, :]
        x_data_sample = np.concatenate([x_data_sample, rand_col], axis=1)
        return x_data_sample

    @staticmethod
    def check_plot_arguments(x_data: np.ndarray,
                             feature_names,
                             sample: float) -> None:

        if sample <= 0 or sample > 1:
            raise ValueError(f"Sample should be in interval (0, 1], but is {sample}")

        if len(feature_names) != x_data.shape[0]:
            raise ValueError(f"We have {len(feature_names)} feature names but shape of "
                             f"input data implies {x_data.shape[0]} features")

    # endregion

    def __init__(self,
                 shape: int,
                 layer_shapes: List[int],
                 random_generator: Optional[np.random.Generator] = None,
                 is_classifier: bool = True,
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
        :param random_generator: numpy random number generator
        :param is_classifier: boolean flag for whether we are building
        a classification or regression model
        :param layer_activation: activation function for the hidden
        layers
        """

        if not layer_shapes:
            raise ValueError("Must provide shape for at least one hidden layer")

        if layer_shapes[-1] != shape:
            raise ValueError(f"Output of final layer should be of size {shape} but "
                             f"is {layer_shapes[-1]}")

        if layer_activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"Activation function {layer_activation} is not supported: "
                             f"valid choices are {', '.join(_VALID_ACTIVATIONS)}")

        if random_generator is None:
            random_generator = np.random.default_rng()

        # model parameters (adjust for random col)
        self._shape = shape + 1  # adjust for random col
        self._layer_shapes = layer_shapes
        self._layer_shapes[-1] += 1  # adjust for random col
        self._is_classifier = is_classifier
        self._layer_activation = layer_activation
        self._rng = random_generator

        # only set after fitting
        self.prediction_model_ = None
        self.beta_model_ = None
        self.conf_ = None

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            val_split: float = 0.1,
            epochs: int = 100,
            use_early_stop: bool = True) -> None:
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

        self.prediction_model_, self.beta_model_ = self._create_keras_model()
        random_col = self._rng.normal(size=(x_train.shape[0], 1))
        x_train = np.concatenate([x_train, random_col])

        callbacks = []
        if use_early_stop:
            early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
            callbacks.append(early_stop)

        self.prediction_model_.fit(x=x_train, y=y_train, epochs=epochs,
                                   validation_split=val_split, callbacks=callbacks)

        betas = self.beta_model_(x_train)
        self.conf_ = np.std(betas[:, -1])

    def plot_betas_by_feature(self,
                              x_data: np.ndarray,
                              feature_names: List[str],
                              sample_size: float = 0.25,
                              cols: int = 3) -> [plt.Figure, plt.Axes]:
        """
        utility function to plot the distribution of the local
        beta features given a set of feature values. Can optionally plot
        confidence intervals to help with feature selection

        :param x_data: data for which we would like to plot our
        beta values
        :param feature_names: names for the features in our x_data
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        :param cols: number of columns in our plot
        """

        # get model betas
        self.check_plot_arguments(x_data, feature_names, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        betas = self.beta_model_(x_data_sample)

        # set up our grid for plotting
        rows = betas.shape[1] // cols
        if betas.shape[1] % cols != 0:
            rows += 1
        fig, axs = plt.subplots(rows, cols)

        # set up our confidence intervals
        conf_intervals = self._get_confidence_intervals()

        # plot scatter chart for each variable
        for i in range(betas.shape[1]):

            r, c = i // cols, i % cols
            ax = axs[r, c]
            ax.scatter(x_data[:, i], betas[:, i])
            ax.set_title(f"Coefficients for {feature_names[i]}")

            for name, definition in conf_intervals.items():
                lower, upper, color = definition
                ax.axhline(lower, color=color, linestyle="--")
                ax.axhline(upper, color=color, linestyle="--")

        return fig, axs

    def plot_interactions(self,
                          x_data: np.ndarray,
                          feature_names: List[str],
                          sample_size: float = 1,
                          cols: int = 2) -> Tuple[plt.Figure, plt.Axes]:
        """
        for each feature we plot a graph showing how the gradients of
        the betas vary with the value of the given feature

        :param x_data: data for which we would like to plot our
        beta values
        :param feature_names: names for the features in our x_data
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        :param cols: number of columns in our plot
        """

        # calculate gradients based on sample
        self.check_plot_arguments(x_data, feature_names, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        input_tensor = tf.convert_to_tensor(x_data_sample)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            beta = self.beta_model_(input_tensor)
        grads = tape.batch_jacobian(beta, input_tensor)
        grads_np = grads.numpy()  # easier to work with numpy array for plotting

        # set up our axes for plotting
        rows = int(math.ceil(x_data.shape[1] / cols))
        fig, axs = plt.subplots(rows, cols)

        # outer loop: plot interactions for each feature
        for i in range(x_data_sample.shape[1]):

            # get axes for plotting
            row, col = i // cols, i % cols
            ax = axs[row, col]
            ax.set_ylim(-0.5, 0.5)  # TODO: change limits based on data
            ax.set_title(f"Interactions for feature {feature_names[i]}", fontsize=15)

            # select feature values and gradients, and reorder
            d_beta_0 = grads_np[:, i, :]
            x_0 = x_data_sample[:, i].values
            order = np.argsort(x_0)
            x_0 = x_0[order]
            d_beta_0 = d_beta_0[order, :]
            x_vals = np.linspace(np.min(x_0), np.max(x_0), 30)

            # inner loop: plot each beta
            for j in range(d_beta_0.shape[1]):
                spline = UnivariateSpline(x_0, d_beta_0[:, j])
                name = f"feature_{j + 1}"
                ax.plot(x_vals, [spline(v) for v in x_vals], label=name)

        # add legend at the figure level
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        return fig, axs

    def plot_feature_importance(self,
                                x_data: np.ndarray,
                                feature_names: List[str],
                                sample_size: float = 1) -> Tuple[plt.Figure, plt.Axes]:
        """
        for each feature we plot a graph showing how the gradients of
        the betas vary with the value of the given feature

        :param x_data: data for which we would like to plot our
        beta values
        :param feature_names: names for the features in our x_data
        :param sample_size: % proportion of the feature values to sample
        to create our plots
        """

        # get betas and calculate feature importance
        self.check_plot_arguments(x_data, feature_names, sample_size)
        x_data_sample = self.get_sampled_data(x_data, sample_size)
        betas = self.beta_model_(x_data_sample)
        avg_abs_betas = abs(betas[:-1]).mean(axis=0)
        importance, threshold = avg_abs_betas[:-1], avg_abs_betas[-1]

        # reorder for plotting
        order = np.argsort(importance)
        importance = importance[order]
        feature_names = [feature_names[i] for i in order]

        # set up axes and make bar plot
        fig, ax = plt.subplots()
        positions = np.arange(x_data.shape[1])
        ax.barh(positions, importance)
        ax.set_yticks(positions)
        ax.set_yticklabels(feature_names)
        ax.axvline(threshold, color="red", linestyle="--")

        return fig, ax
