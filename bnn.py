import tensorflow as tf
from activations import *
import seaborn as sbn
import boto
from kiwisolver import Solver
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import time
import sys
import ast
from z3 import *
from larq.layers import QuantDense
from tensorflow.keras.layers import  Lambda
from tensorflow.python.framework import ops


@tf.custom_gradient
def binairy_STE_after_sigmoid(x):
    def grad(dy):
        return dy

    result = tf.round(x)
    return result, grad 
class Binarizer:
    @staticmethod
    def binarize(x):
        return tf.where(tf.math.greater_equal(x, 0), 1.0, -1.0)

class LinearLayer(QuantDense):
    def __init__(self, num_neurons, activation=None):
        super().__init__(num_neurons)
        self.num_neurons = num_neurons
        self.activation=activation
        self.dense_layer=tf.keras.layers.Dense(units=num_neurons,activation=activation,kernel_initializer='glorot_uniform')
        self.prob=tf.constant(0.5)


    def binary_stochastic(self,x):
        binary_mask=tf.cast(tf.random.uniform(tf.shape(x)) < self.prob,dtype=tf.float32)
        output=x*binary_mask+ tf.stop_gradient(tf.sign(x)-x)*(1-binary_mask)

        return output
    @tf.custom_gradient
    def ste_activation (self, x):
        def grad(dy):
            return dy
        return self.binary_stochastic(x), grad

    

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer=self.binarized_initializer,
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='glorot_uniform',
                                    trainable=True)
    def binarize(self,x):
        return tf.where(tf.math.greater_equal(x, 0), 1.0, -1.0)
    def binarize_ste(self, x):
            """
            Clip and binarize tensor using the straight through estimator (STE) for the gradient.
            """

            with ops.name_scope("Binarized") as name:
                #x=tf.clip_by_value(x,-1,1)
                    return tf.sign(x)
    def binarized_initializer(self, shape, dtype=None, seed=32):
        tf.random.set_seed(seed)
        # Initialize weights using Binarizer
        return Binarizer.binarize(tf.random.normal(shape, dtype=dtype))


    def call(self, inputs):
        # Output of the original layer
        #print('LinearLayer is called...')
        #binarized_inputs = Binarizer.binarize(inputs)
        #binarized_weights= Binarizer.binarize(self.W)
        #output = tf.matmul(binarized_inputs, binarized_weights) + self.bias
        #tf.print('Weights LinearLayer',binarized_weights[0])
        #output=self.dense_layer(inputs)
        #output_ste=self.ste_activation(output)
        #self.W = tf.clip_by_value(self.W,-1,1)
        binarized_weights = self.binarize_ste(tf.clip_by_value(self.W,-1,1))
        binarized_inputs = self.binarize_ste(inputs)
        output = tf.matmul(binarized_inputs, binarized_weights) + self.bias
        

        """
        z3_inputs = [Real(f'input_{i}') for i in range(inputs.shape[-1])]
        z3_weights = [Real(f'weight_{i}') for i in range(self.W.shape[-1])]
        z3_bias = Real('bias')
        z3_constraint = Sum([z3_inputs[i] * z3_weights[i] for i in range(len(z3_inputs))]) + z3_bias >= 0

        # Create a Z3 solver
        solver = Solver()

        # Add the constraint to the solver
        solver.add(z3_constraint)

        # Set the values from the Keras layer to the Z3 variables
        for i in range(inputs.shape[-1]):
            solver.add(z3_inputs[i] == inputs[0, i])

        for i in range(self.W.shape[-1]):
            solver.add(z3_weights[i] == self.W[i])

        solver.add(z3_bias == self.bias)

        # Check if the constraint is satisfiable
        result = solver.check()

        if result == sat:
            print("Constraint is satisfied during forward pass.")
        else:
            print("Constraint is violated during forward pass.")"""
        #tf.print('Outputs LinearLayer',output)

        return output


class BatchNormLayer(QuantDense):
    def __init__(self, num_neurons, epsilon=1e-8):
        super().__init__(num_neurons)
        self.num_neurons = num_neurons
        self.epsilon = epsilon
        

    def build(self, input_shape):
        # Initialize the weights and bias
        self.W = self.add_weight(shape=(input_shape[-1], self.num_neurons),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.num_neurons,),
                                    initializer='glorot_uniform',
                                    trainable=True)

    def call(self, inputs):
        #print('BatchNormLayer is called...')
        mean, var = tf.nn.moments(inputs, axes=[0])
        std = tf.sqrt(var + self.epsilon)

        normalized_inputs = (inputs - mean) / std

        output = tf.matmul(normalized_inputs, self.W) + self.bias
        #tf.print('Output Batch ',output)

        return output


class BINLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def binarize(self, z):
        output = tf.where(tf.math.greater_equal(z, 0), 1.0, -1.0)
        #tf.print('Output BIN ',output)
        return output

    def z3_constraint(self, inputs, v):
        # Define symbolic variable
        constraint_inputs = Real('constraint_inputs')

        # Constraint: v should be equal to 1 when inputs is greater than 0
        constraint = Implies(constraint_inputs > 0, v == 1)

        # Check satisfiability
        solver = Solver()
        solver.add(constraint)
        solver.add(inputs == constraint_inputs)


        z3_result = solver.check()

        if z3_result == sat:
            print("Z3 Constraint is satisfiable. v equals 1 when inputs is greater than 0.")
            # Print the Z3 model
            print(solver.model())
        elif z3_result == unsat:
            print("Z3 Constraint is unsatisfiable. v does not equal 1 when inputs is greater than 0.")
        else:
            print("Z3 Constraint result is unknown.")

        return solver

    def call(self, inputs):
        v = self.binarize(inputs)
        #tf.print('v shape: ', v.shape)

        # Check the Z3 constraint
        #inputs_np = tf.keras.backend.get_value(inputs).numpy()
        #print('inputs_np',inputs_np)
        

        return v

class SigmoidLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        #print('SigmoidLayer is called...')
        return tf.math.sigmoid(inputs)
    

class BNN(tf.keras.Sequential):
    def __init__(self, input_dim, output_dim, num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=16):
        super(BNN, self).__init__()

        self.add(layers.InputLayer(input_shape=(input_dim,)))
        self.add(LinearLayer(num_neuron_in_hidden_dense_layer, activation=binary_sigmoid))
        self.add(BatchNormLayer(num_neuron_in_hidden_dense_layer))
        self.add(BINLayer())
        self.add(LinearLayer(num_neuron_in_hidden_dense_layer,activation=binary_sigmoid))
        self.add(BatchNormLayer(num_neuron_in_hidden_dense_layer))
        self.add(BINLayer())
        self.add(LinearLayer(num_neuron_output_layer,activation=binary_sigmoid))
        self.add(SigmoidLayer())
        self.add(Lambda(binairy_STE_after_sigmoid))

    def print_linear_layer_weights(self):
            print('In Model ')
            for layer in self.layers:
                if isinstance(layer, LinearLayer):
                    weights, biases = layer.get_weights()
                    print(f"Weights for LinearLayer: {weights}")
                    print(f"Biases for LinearLayer: {biases}")
