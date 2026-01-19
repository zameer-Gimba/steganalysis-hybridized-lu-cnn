import unittest
import tensorflow as tf
from src.hybrid_lu_cnn import build_sample_cnn, hybridize_lu_activation_function_in_cnn

class HybridLUActivationTest(unittest.TestCase):

    def setUp(self):
        self.model = build_sample_cnn()
        self.modified_model = hybridize_lu_activation_function_in_cnn(self.model)

    def test_model_structure_preserved(self):
        """Ensure layer count remains unchanged after hybridization."""
        self.assertEqual(len(self.model.layers), len(self.modified_model.layers))

    def test_model_is_keras_model(self):
        """Check returned object is a valid Keras model."""
        self.assertIsInstance(self.modified_model, tf.keras.Model)

    def test_activation_layers_exist(self):
        """Ensure activation layers still exist after modification."""
        activation_layers = [
            layer for layer in self.modified_model.layers
            if isinstance(layer, tf.keras.layers.Activation)
        ]
        self.assertTrue(len(activation_layers) >= 1)

if __name__ == "__main__":
    unittest.main()
