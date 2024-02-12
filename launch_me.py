from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from larq.layers import QuantDense
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
from pysdd.sdd import SddManager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
import larq as larq
from bnn import *
from cnf import *
from bnn_to_cnf import*
import math
from pysat.solvers import Glucose3
from pysat.formula import CNF
import itertools
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import  optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Layer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
def get_variable_info(cnf_clauses):
    variable_info = {}

    for clause in cnf_clauses.clauses:
        for term in clause.terms:
            var_name = term.annotation
            variable_info[var_name] = {
                'type': 'original' if 'in' in var_name else 'auxiliary',
                'literal': term.tid
            }
    #print('Actual content variable_info', variable_info)

    return variable_info
def check_variable_assignment(cnf_file):
    solver = Glucose3()
    cnf_formula = CNF(from_file=cnf_file)
    print('CNF clauses :', type(cnf_formula.clauses[0][0]))
    cnf_clauses = Cnf([])
    for cl in cnf_formula.clauses:
        cnf_clauses+=cl

    # Solve the CNF formula
    for c in cnf_clauses:
         solver.add_clause(c)
    print(solver.solve())
    if solver.solve():
        model = solver.get_model()
        #print('cnf content', cnf_formula)
        variable_info = get_variable_info(cnf_clauses)
        print('variable_info',variable_info)
        
        # Check if each variable corresponds to the correct literal
        for var_name, info in variable_info.items():
            expected_literal = info['literal']
            print('expected_literal', expected_literal)
            actual_value = model[abs(expected_literal) - 1] * sign(expected_literal)

            print(f"Variable: {var_name}, Expected Literal: {expected_literal}, Actual Value: {actual_value}")

    else:
        print("The CNF formula is unsatisfiable.")
"""
 def write_variable_info(variable_info, filename):
    # Take only the last 20 variable names, types, and literals
    last_20_variable_info = list(variable_info.items())[-500:]
    print(list(variable_info.items()))

    with open(filename, 'w') as f:
        f.write('c.vars: {}\n'.format(' '.join(var_name for var_name, _ in last_20_variable_info)))
        f.write('c.var_types: {}\n'.format(' '.join(info['type'] for _, info in last_20_variable_info)))
        f.write('c.var_literals: {}\n'.format(' '.join(str(info['literal']) for _, info in last_20_variable_info)))
        f.write('\n')"""

def write_variable_info(variable_info, filename):
    # Take only the last 20 variable info entries
    last_20_variable_info = list(variable_info.items())

    with open(filename, 'w') as f:
        original_variable_info = [(var_name, info) for var_name, info in last_20_variable_info if info['type'] == 'original']
        
        f.write('c.vars: {}\n'.format(' '.join(var_name for var_name, _ in original_variable_info)))
        f.write('c.var_types: {}\n'.format(' '.join(info['type'] for _, info in original_variable_info)))
        f.write('c.var_literals: {}\n'.format(' '.join(str(info['literal']) for _, info in original_variable_info)))
        f.write('\n')




def read_and_print_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
def get_num_variables_and_clauses_from_cnf(cnf_filename):
    num_variables, num_clauses = None, None
    
    with open(cnf_filename, 'r') as cnf_file:
        for line in cnf_file:
           
            if line.startswith('p cnf'):
               
                _, _, num_variables, num_clauses = line.split()
                num_variables, num_clauses = int(num_variables), int(num_clauses)
                break  
    print('Num Vars :', num_variables)
    print('num clauses :', num_clauses)
                
    return num_variables, num_clauses
def replace_question_mark(sequence):
    return [1 if bit == '?' else int(bit) for bit in sequence]
def write_output(cnf_array, num_terms, filename):
    with open(filename, 'a') as f:
        f.write('p cnf %d %d\n' % (num_terms, len(cnf_array)))
        for clause in cnf_array:
            f.write(' '.join(map(str, clause)) + ' 0\n')
    print("Wrote cnf to %s" % filename)

if __name__ == "__main__":
     loaded_dataset = pd.read_csv('mydata.csv', dtype='str')

    
     X = loaded_dataset['X'].apply(lambda x: list(map(int, x)))
     Y = loaded_dataset['Y'].apply(lambda y: list(map(int, y)))

    
     X = np.array([np.array(x) for x in X])
     Y = np.array([np.array(y) for y in Y])
     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_train)
     X_test_scaled = scaler.transform(X_test)
     opt = larq.optimizers.Bop(threshold=1e-02, gamma=0.01, name="Bop")
     model = BNN(num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=4, input_dim=18, output_dim=4)
     #model.print_linear_layer_weights()
     #linear_layer_weights = model.layers[6].get_params_weights()
     #print(type(model.layers[6]))
     #print("Linear Layer Weights:", linear_layer_weights)
     #print("Linear Layer Biases:", linear_layer_biases)
     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
     #initial_weights = model.get_weights()
     #model.save_weights("initial_weights.h5")

     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
     model.fit(X_train_scaled, Y_train, epochs=80, batch_size=80, validation_split=0.2, callbacks=[early_stopping])
     

    # Generate some test data
     Y_true_numeric = Y_test
     X_test_numeric = X_test_scaled

     Y_pred = model.predict(X_test_numeric)

     Y_pred_binary = [''.join(map(str, np.round(pred))) for pred in Y_pred]
     evaluation = model.evaluate(X_test_scaled, Y_test)
     test_accuracy = evaluation[1]
     print(f'Test Accuracy: {test_accuracy}')


     print(model.layers[0].get_weights()[0][0])
     count=0
     for i in range(len(Y_true_numeric)):
        true_str = ''.join(map(str, Y_true_numeric[i]))
        pred_str = ''.join(map(lambda x: str(int(x)), np.round(Y_pred[i])))
        if true_str==pred_str: count=count+1

        #print(f"True: {true_str}, Predicted: {pred_str}")
     #print(f'out {count} well predicted  of {len(Y_true_numeric)} ')

     datafile = 'weights_after_training.h5'

     #model.save_weights(datafile)

     cnf_clauses = Cnf([])
     
     #dim= [ [18,18], [18,18],[18,18],[18,4],[4,] ]
     input_dim=18
     num_neuron_in_hidden_dense_layer=18
     num_neuron_output_layer=4
     dim = [
    (input_dim, num_neuron_in_hidden_dense_layer),
    (num_neuron_in_hidden_dense_layer, num_neuron_in_hidden_dense_layer),
    (num_neuron_in_hidden_dense_layer, num_neuron_in_hidden_dense_layer),
    (num_neuron_in_hidden_dense_layer, num_neuron_output_layer)
]

     print('################################################### Encoding Process starts... ###########################################################################')

     #linear_layer_weights, linear_layer_biases = bnn_model.get_layer('linear_layer_name').get_weights()
     #print("Linear Layer Weights:", linear_layer_weights)
     #print("Linear Layer Biases:", linear_layer_biases)

     input_terms = [Term(annotation='in' + str(i)) for i in range(dim[0][0])]
     output_terms = input_terms
     print('NUM layers :', len(model.layers))

     for layer, layer_id in zip(model.layers, range(len(model.layers))):
        if isinstance(layer, LinearLayer):
            if layer_id != len(model.layers)-1 :
                #x = layer.get_input()
                weights_in = layer.get_weights()[0]  # Assuming the weights are the first element in the list returned by get_weights
                biases_in = layer.get_weights()[1]
                #print('layer.get_weights()[0]:', weights_in[0].shape)
                #print('layer.get_weights()[1]:', biases[:5])
                print('Internal :', layer.name)


                (output_terms, output_clauses) = internal_layer_to_cnf(output_terms, weights_in.T, biases_in, 'L'+ str(layer_id))
                #print(f'output_terms Internal: {output_terms[0]}')
               
            else:
                #input_terms = [Term(annotation='out' + str(i)) for i in range(dim[-1][0])]
                #output_terms= input_terms
                weights_out = layer.get_weights()[0]  # Assuming the weights are the first element in the list returned by get_weights
                biases_out = layer.get_weights()[1]
                print('layer_id :',layer_id)

                print('External :', layer.name)
                #print('weights LAST :', weights_out.T)
                #print('weights ALL :', weights_out)

                (output_terms, output_clauses) = output_layer_to_cnf(output_terms, weights_out.T, biases_out, 'L' + str(layer_id))
                print(f'output_clauses External: {type(output_clauses)}')

            
            cnf_clauses += output_clauses

     variable_info= get_variable_info(cnf_clauses)

     s = set()
     d = {}
     for clause in cnf_clauses.clauses:
        #print((clause))
        for term in clause.terms:
            s.add(abs(term.tid))

     sorted_s = sorted(s)
     for i, tid in enumerate(sorted_s):
        d[tid] = i + 1

     cnf_array = []
     for clause in cnf_clauses.clauses:
        clause_array = []
        for term in clause.terms:
            clause_array.append(d[abs(term.tid)] * sign(term.tid))

        cnf_array.append(clause_array)

     swap0 = dim[0][0] + 1
     #for layer_dimensions in dim[1:]:
        #swap0 += layer_dimensions[0] + 1

     #print('swap0 :',swap0)
     swap1 = d[abs(output_terms[0].tid)]
     for i, clause in enumerate(cnf_array):
        #print(cnf_array)
        for j, term in enumerate(clause):
            if abs(term) == swap0:
                cnf_array[i][j] = swap1 * sign(term)
            elif abs(term) == swap1:
                cnf_array[i][j] = swap0 * sign(term)

     print("swapped %d with %d" % (swap0, swap1))
     output_file='bnntocnf.cnf'
     write_variable_info(variable_info, output_file)
     write_output(cnf_array, len(s), output_file)
     #check_variable_assignment(output_file)

     print('################################################### End of the Encoding Process !!!  ###########################################################################')


     

     #model.save("model.h5")
     larq.models.summary(model)
     #describe_network(model)
     plt.figure(figsize=(12, 6)) 

    # Plot training & validation accuracy values
     """
     plt.subplot(1, 2, 1)
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.title('Model accuracy')
     plt.ylabel('Accuracy')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
     plt.subplot(1, 2, 2)
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('Model loss')
     plt.ylabel('Loss')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

     plt.tight_layout()
     plt.show()"""