import unittest
import numpy as np
import numpy.random
from local_glm_net import LocalGlmNet


class TestLocalGlmNet(unittest.TestCase):

    @classmethod
    def _get_glm_model(cls):
        rng = np.random.default_rng(1234)
        model = LocalGlmNet(4, [2, 4], "regression", rng)
        return model

    def test_feature_to_plot_not_in_col_indices(self):
        # arrange
        model = self._get_glm_model()
        test_features_to_plot = ["name_1", "name_2", "name_3"]
        model.col_indices_ = {"name_1": 1, "name_3": 2}
        test_sample = 0.5

        # act/assert
        with self.assertRaises(ValueError):
            model.check_plot_arguments(test_features_to_plot, test_sample)

    def test_categorical_feature_not_in_col_indices(self):
        # arrange
        model = self._get_glm_model()
        test_features_to_plot = ["feature1", "feature2"]
        model.col_indices_ = {"feature1_val1": 1, "feature1_val2": 2}
        test_sample = 0.5

        # act/assert
        with self.assertRaises(ValueError):
            model.check_plot_arguments(test_features_to_plot, test_sample,
                                       is_categorical=True)

    def test_sample_negative_fails(self):
        # arrange
        model = self._get_glm_model()
        test_data = np.zeros(shape=(2, 2))
        test_feature_names = ["name_1", "name_2", "name_3"]
        model.col_indices_ = {"name_1": 1, "name_2": 2, "name_3": 3}
        test_sample = -0.1

        # act/assert
        with self.assertRaises(ValueError):
            model.check_plot_arguments(test_feature_names, test_sample)

    def test_sample_bigger_than_one_fails(self):
        # arrange
        model = self._get_glm_model()
        test_data = np.zeros(shape=(2, 2))
        test_feature_names = ["name_1", "name_2", "name_3"]
        model.col_indices_ = {"name_1": 1, "name_2": 2, "name_3": 3}
        test_sample = -0.1

        # act/assert
        with self.assertRaises(ValueError):
            model.check_plot_arguments(test_feature_names, test_sample)

    def test_get_sampled_data(self):
        # arrange
        test_data = np.array([[0.1, -0.2, 0.3],
                              [0.0, 0.1, -0.4],
                              [-0.3, 0.1, 0.7],
                              [0.4, 0.4, -0.2]])
        test_sample = 0.75
        model = self._get_glm_model()

        # act
        result = model.get_sampled_data(test_data, test_sample)

        # assert
        expected_result = np.array([[0.0, 0.1, -0.4, 0.1526191],
                                    [-0.3, 0.1, 0.7, 0.863744],
                                    [0.4, 0.4, -0.2, 2.913099]])
        np.testing.assert_array_almost_equal(expected_result, result)

    def test_get_confidence_intervals(self):
        # arrange
        model = self._get_glm_model()
        model.conf_ = 0.4

        # act
        result = model._get_confidence_intervals()

        # assert
        expected_result = {"95%": (-0.784, 0.784, "yellow"),
                           "99%": (-1.1228, 1.1228, "orange"),
                           "99.9%": (-1.3164, 1.3164, "red")}

        self.assertDictEqual(expected_result, result)
