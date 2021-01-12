import unittest

import numpy as np

import functions
import neuralnet
import utils


class TestFunctions(unittest.TestCase):
    """test class of functions.py"""

    def test_relu(self):
        test_data = np.array([[1, 0, -0.1], [-0.5, -0.001, 0.001]])
        expected = np.array([[1, 0, 0], [0, 0, 0.001]])
        actual = functions.relu(test_data)
        self.assertTrue((expected == actual).all())

    def test_relu_derivative(self):
        test_data = np.array([[1, 0, -0.1], [-0.5, -0.001, 0.001]])
        expected = np.array([[1, 0, 0], [0, 0, 1]])
        actual = functions.relu_derivative(test_data)
        self.assertTrue((expected == actual).all())

    def test_softmax(self):
        test_data = np.array([[0.3, 2.9, 4.0], [-0.5, -0.001, 0.001]])
        expected = np.array([[1.0, 1.0]])

        actual = functions.softmax(test_data)
        print(actual)
        actual = np.sum(actual, axis=1)
        print(actual)
        self.assertTrue((expected == actual).all())

    def test_softmax_1(self):
        test_data = np.array([[0.3, 2.9, 4.0], [-0.5, -0.001, 0.001]])
        expected = (2, 3)
        logits = functions.softmax(test_data)
        print(logits)
        actual = logits.shape
        self.assertTrue(expected == actual)


class TestNeuralnet(unittest.TestCase):
    def test_cross_entropy_loss(self):
        model = neuralnet.MLP_MNIST(2, 2, 2, 2)

        test_y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        test_y_hat = np.array([0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
        actual = model.cross_entropy_loss(test_y, test_y_hat)
        print(actual)
        estimate = 0.51082545
        self.assertTrue(abs(actual - estimate) < 0.001)

    def test_cross_entropy_loss_1(self):
        model = neuralnet.MLP_MNIST(2, 2, 2, 2)

        test_y = np.array(
            [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
        )
        test_y_hat = np.array(
            [
                [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
                [0.1, 0.005, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
            ]
        )
        actual = model.cross_entropy_loss(test_y, test_y_hat)
        print(actual)
        estimate = 0.51082545
        self.assertTrue(abs(actual - estimate) < 0.001)


if __name__ == "__main__":

    # load dataset
    dataset = utils.mnist_dataset("data")
    X_train, X_test, y_train, y_test = dataset.load()
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)

    # Test of the generator
    batch_generator = utils.batch_generator(
        X_train, y_train, batch_size=3, shuffle=True
    )
    print("-----Test generator------")
    print(next(batch_generator))
    print(next(batch_generator))

    h1_nodes = 256
    h2_nodes = 128
    model = neuralnet.MLP_MNIST(
        input_nodes=784, h1_nodes=h1_nodes, h2_nodes=h2_nodes, output_nodes=10
    )
    X, y = next(batch_generator)
    print("-----Test predict------")
    print(model.predict(X))
    print("-----Test forward------")
    print(model.forwardpass_train(X))

    ## UnitTest
    unittest.main()
