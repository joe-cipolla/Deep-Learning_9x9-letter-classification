# -*- coding: utf-8 -*-
import random
from math import exp
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class Letter(object):
    """A class to hold one instance of a given letter.
    """

    def __init__(self, flattened_letter_array, target_letter):
        self.array = flattened_letter_array
        self.letter = target_letter
        self.target = ord(self.letter.lower()) - 97


def load_letters(size=81, use_variants=0):
    """ Load all letters of given size (25, 81, etc)

    :param size:
    :param use_variants: boolean indicating to include variant letters or not
    :return: number of unique letters and a list of Letter objects
    """
    from os import listdir
    basedir = '/Users/joecipolla/Documents/Reference/Education/490-DL_Deep Learning/Week_4/'

    if size == 25:
        alpha_dir = basedir + '/5x5_alphabet/'
    elif size == 81:
        alpha_dir = basedir + '/9x9_alphabet/'

    letters = [f for f in listdir(alpha_dir) if f[0] != '.']
    letter_objects = []
    for letter in letters:
        letter_dir = alpha_dir + letter + '/'
        for instance in [f for f in listdir(letter_dir)]:
            first_instance = str(letter) + "_0.letter"
            if (use_variants == 0) and (str(instance) != first_instance):
                pass
            else:
                instance_array = []
                with open(letter_dir + instance) as f:
                    for line in f:
                        if '#' not in line:
                            instance_array.extend(line.replace(' ', '').replace(',', '').replace('\n', ''))
                letter_objects.append(Letter([int(x) for x in instance_array], letter))

    return len(letters), letter_objects


def add_noise(all_letters, noise_percent=0, num_noisy_nodes=0):
    i = 0
    while i < (num_noisy_nodes - 1):
        letter = random.choice(all_letters)
        letter_array = letter.array
        p = 0
        while (p / float(len(letter_array))) < noise_percent:
            letter_array[random.choice(range(len(letter_array)))] = random.choice(range(2))
            p += 1
        letter.array = letter_array
        i += 1
    return all_letters


def get_one_example_per_letter(all_letters):
    letters = set([l.letter for l in all_letters])
    letter_objects = []
    while len(letters) > 0:
        let = random.choice(all_letters)
        if let.letter in letters:
            letter_objects.append(let)
            letters.remove(let.letter)
    return letter_objects


def welcome():
    print
    print '******************************************************************************'
    print
    print 'Welcome to the Multilayer Perceptron Neural Network'
    print '  trained using the backpropagation method.'
    print 'Version 1.0, 10/20/2017, Joe Cipolla'
    print 'For comments, questions, or bug-fixes, contact: jlcipolla@gmail.com'
    print ' '
    print '******************************************************************************'
    print
    return ()


# The 'transfer' function  -- From Derek
def transfer(x, alpha=1.0, deriv=False, method="sigmoid"):
    if method == "sigmoid":
        if deriv == True:
            return alpha * x * (1.0 - x)
        else:
            return 1.0 / (1.0 + np.exp(-alpha * x))
    elif method == "tanh":
        if deriv == True:
            return alpha * (1.0 - np.tanh(alpha * x) ** 2)
        else:
            return np.tanh(alpha * x)
    elif method == "ReLU":
        if deriv == True:
            return alpha * np.exp(alpha * x) / (1 + np.exp(alpha * x))
        else:
            return np.log(1 + np.exp(alpha * x))


def sigmoid_transfer(summed_neuron_input, alpha):
    activation = 1.0 / (1.0 + exp(-alpha * summed_neuron_input))
    return activation


def ReLU_transfer(summed_neuron_input, alpha):
    activation = alpha * np.maximum(summed_neuron_input, 0)
    return activation


def differentiate_transfer(neuron_output, alpha):
    """ Backpropagation transfer function """
    return alpha * neuron_output * (1.0 - neuron_output)


def dot_product(matrix1, matrix2):
    try:
        dot_prod = np.dot(matrix1, matrix2)
    except:
        dot_prod = np.dot(np.transpose(matrix1), matrix2)
    return dot_prod


def get_random_weight():
    return 1 - 2 * random.random()


def initialize_weights(dimensions):
    """Take in a dimension for an np.array and create an np.array, filled with random weights.
    :param dimensions: a list defining the dimensions of the array we wish to create.
    :return: a numpy array filled with random weights.
    """
    weights = np.zeros(dimensions)
    for weight in np.nditer(weights, op_flags=['readwrite']):
        weight[...] = get_random_weight()
    return weights


def initialize_bias_weights(num_bias_nodes):
    bias_weights = np.zeros(num_bias_nodes)
    for node in range(num_bias_nodes):
        bias_weights[node] = get_random_weight()
    return bias_weights


# def obtain_selected_alphabet_training_values(training_data_set_num):
#       *** need to complete ***
#   ''' allows you to test single letter - and all its variants '''


def feed_forward(alpha, inputs, weights, biases):
    """Compute feed forward outputs

    :param alpha:
    :param inputs:
    :param weights:
    :param biases:
    :return: An array of output values from the hidden layer. Used as inputs to the output layer.
    """
    activations = dot_product(weights, inputs) + biases
    output = np.zeros(np.shape(activations))

    for node in range(len(output)):
        output[node] = sigmoid_transfer(activations[node], alpha)

    return output


def compute_training_outputs(alpha, num_training_sets, w_weights, hidden_biases, v_weights, output_biases,
                             letters=list()):
    letter = 0
    training_results = {}

    while letter < num_training_sets:
        input_values = letters[letter].array
        # Feed forward inputs through hidden layer to predicted outputs
        hidden_values = feed_forward(alpha, input_values, w_weights, hidden_biases)
        output_values = feed_forward(alpha, hidden_values, v_weights, output_biases)

        desired_class = letters[letter].target  # identify the desired class
        desired_letter = letters[letter].letter
        desired_output = np.zeros(len(output_values))
        desired_output[desired_class] = 1  # set the desired output for that class to 1

        # Calc SSE - difference between predicted and actual
        output_errors = np.zeros(len(output_values))
        new_sse = 0.0
        for node in range(len(output_values)):  # Number of nodes in output set (classes)
            output_errors[node] = desired_output[node] - output_values[node]
            new_sse = new_sse + output_errors[node] * output_errors[node]

        letter += 1
        training_results[desired_letter] = {'hidden': hidden_values.tolist(), 'output': output_values.tolist(),
                                            'desired_output': desired_output.tolist(), 'errors': output_errors.tolist(),
                                            'new_sse': new_sse}

    return training_results


########################################################################################################################
#   BACKPROPAGATION FUNCTIONS
########################################################################################################################

# Backpropagate weight changes onto the hidden-to-output connection weights
def backprop_output_to_hidden(alpha, eta, output_errors, output_values, hidden_values, v_weights):
    """
    The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    Core equation for the first part of backpropagation:
    d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    where:
    -- SSE = sum of squared errors, and only the error associated with a given output node counts
    -- v(h,o) is the connection weight v between the hidden node h and the output node o
    -- alpha is the scaling term within the transfer function, often set to 1
    ---- (this is included in transfFuncDeriv)
    -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    -- F = transfer function, here using the sigmoid transfer function
    -- Hidden(h) = the output of hidden node h.

    We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
      of the SSE w/r/t the weight v.
    This means, since there is a minus sign in that derivative, that we will add a small amount.
    (Decrementing is -, applied to a (-), which yields a positive.)
    """

    # Unpack array lengths
    num_hidden_nodes = len(hidden_values)
    num_output_nodes = len(output_values)

    transfer_partial_deriv = np.zeros(num_output_nodes)  # initialize an array for the transfer function

    for node in range(num_output_nodes):  # Number of hidden nodes
        transfer_partial_deriv[node] = differentiate_transfer(output_values[node], alpha)

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in the equations for the deltas in the connection weights

    delta_v_wt_array = np.zeros((num_output_nodes, num_hidden_nodes))  # initialize an array for the deltas
    new_v_weights = np.zeros((num_output_nodes, num_hidden_nodes))  # init array for new hidden weights

    for row in range(num_output_nodes):  # Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(num_hidden_nodes):  # number of columns in weightMatrix
            partial_sse_w_v_wt = -output_errors[row] * transfer_partial_deriv[row] * hidden_values[col]
            delta_v_wt_array[row, col] = -eta * partial_sse_w_v_wt
            new_v_weights[row, col] = v_weights[row, col] + delta_v_wt_array[row, col]

    return new_v_weights


# Backpropagate weight changes onto the bias-to-output connection weights
def backprop_output_biases(alpha, eta, output_errors, output_values, output_biases):
    """
    The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    Core equation for the first part of backpropagation:
    d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    where:
    -- SSE = sum of squared errors, and only the error associated with a given output node counts
    -- v(h,o) is the connection weight v between the hidden node h and the output node o
    -- alpha is the scaling term within the transfer function, often set to 1 (this is included in transfFuncDeriv)
    -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    -- F = transfer function, here using the sigmoid transfer function
    -- Hidden(h) = the output of hidden node h.

    Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
        scales amount of change to connection weight

    We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
        of the SSE w/r/t the weight biasOutput(o).
    This means, since there is a minus sign in that derivative, that we will add a small amount.
    (Decrementing is -, applied to a (-), which yields a positive.)

    Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
        and so is not included explicitly in these equations

    The equation for the actual dependence of the Summed Squared Error on a given bias-to-output
        weight biasOutput(o) is:
        partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
    The transfer function derivative (transFuncDeriv) returned from differentiate_transfer is given as:
        transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
    Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
        partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
        The parameter alpha is included in transFuncDeriv
    """

    # Unpack the output array length
    num_output_nodes = len(output_values)

    delta_output_biases = np.zeros(num_output_nodes)
    new_output_biases = np.zeros(num_output_nodes)
    transfer_partial_deriv = np.zeros(num_output_nodes)

    for node in range(num_output_nodes):  # Number of hidden nodes
        transfer_partial_deriv[node] = differentiate_transfer(output_values[node], alpha)

    for node in range(num_output_nodes):  # Number of nodes in output array (same as number of output bias nodes)
        partial_sse_w_bias_output = -output_errors[node] * transfer_partial_deriv[node]
        delta_output_biases[node] = -eta * partial_sse_w_bias_output
        new_output_biases[node] = output_biases[node] + delta_output_biases[node]

    return new_output_biases


# Backpropagate weight changes onto the input-to-hidden connection weights
def backprop_hidden_to_input(alpha, eta, output_errors, output_values, hidden_values, input_values, v_weights,
                             w_weights):
    """
   The first step here applies a backpropagation-based weight change to the input-to-hidden wts w.

    Core equation for the second part of backpropagation:
    d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
    where:
    -- SSE = sum of squared errors, and only the error associated with a given output node counts
    -- w(i,h) is the connection weight w between the input node i and the hidden node h
    -- v(h,o) is the connection weight v between the hidden node h and the output node o
    -- alpha is the scaling term within the transfer function, often set to 1 (this is included in transfFuncDeriv)
    -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    -- F = transfer function, here using the sigmoid transfer function
            NOTE: in this second step, the transfer function is applied to the output of the hidden node,
            so that F = F(h)
    -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function).
    -- Input(i) = the input at node i.
    -- eta = training rate; scales amount of change to connection weight

    We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
      of the SSE w/r/t the weight w.
    This means, since there is a minus sign in that derivative, that we will add a small amount.
    (Decrementing is -, applied to a (-), which yields a positive.)

    Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
      scales amount of change to connection weight

    For the second step in backpropagation (computing deltas on the input-to-hidden weights)
      we need the transfer function derivative is applied to the output at the hidden node
    """

    num_input_nodes = len(input_values)
    num_hidden_nodes = len(hidden_values)
    num_output_nodes = len(output_values)

    hidden_transfer_derivative = np.zeros(num_hidden_nodes)
    error_times_transfer_deriv = np.zeros(num_output_nodes)
    output_transfer_derivative = np.zeros(num_output_nodes)
    weighted_output_errors = np.zeros(num_hidden_nodes)

    for node in range(num_hidden_nodes):
        hidden_transfer_derivative[node] = differentiate_transfer(hidden_values[node], alpha)

    for node in range(num_output_nodes):
        output_transfer_derivative[node] = differentiate_transfer(output_values[node], alpha)
        error_times_transfer_deriv[node] = output_errors[node] * output_transfer_derivative[node]

    for hidden_node in range(num_hidden_nodes):
        weighted_output_errors[hidden_node] = 0
        for output_node in range(num_output_nodes):
            weighted_output_errors[hidden_node] += v_weights[output_node, hidden_node] * \
                                                   error_times_transfer_deriv[output_node]

    delta_w_weights = np.zeros((num_hidden_nodes, num_input_nodes))
    new_w_weights = np.zeros((num_hidden_nodes, num_input_nodes))

    for row in range(num_hidden_nodes):
        for col in range(num_input_nodes):
            partial_sse_w_weights = -hidden_transfer_derivative[row] * input_values[col] * weighted_output_errors[row]
            delta_w_weights[row, col] = -eta * partial_sse_w_weights
            new_w_weights[row, col] = w_weights[row, col] + delta_w_weights[row, col]

    return new_w_weights


# Backpropagate weight changes onto the bias-to-hidden connection weights
def backprop_hidden_biases(alpha, eta, output_errors, output_values, hidden_values, v_weights, hidden_biases):
    """
    The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    Core equation for the first part of backpropagation:
    d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    where:
    -- SSE = sum of squared errors, and only the error associated with a given output node counts
    -- v(h,o) is the connection weight v between the hidden node h and the output node o
    -- alpha is the scaling term within the transfer function, often set to 1
    ---- (this is included in transfFuncDeriv)
    -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    -- F = transfer function, here using the sigmoid transfer function
    -- Hidden(h) = the output of hidden node h.

    Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
      scales amount of change to connection weight

    We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
      of the SSE w/r/t the weight biasOutput(o).
    This means, since there is a minus sign in that derivative, that we will add a small amount.
    (Decrementing is -, applied to a (-), which yields a positive.)
    """

    num_hidden_nodes = len(hidden_values)
    num_output_nodes = len(output_values)

    # Compute the transfer function derivatives as a function of the output nodes.
    # Note: As this is being done after the call to the backpropagation on the hidden-to-output weights,
    #   the transfer function derivative computed there could have been used here; the calculations are
    #   being redone here only to maintain module independence

    error_times_transfer_deriv = np.zeros(num_output_nodes)
    output_transfer_derivative = np.zeros(num_output_nodes)
    weighted_output_errors = np.zeros(num_hidden_nodes)

    hidden_transfer_derivative = np.zeros(num_hidden_nodes)
    partial_sse_w_bias_hidden = np.zeros(num_hidden_nodes)

    for node in range(num_hidden_nodes):
        hidden_transfer_derivative[node] = differentiate_transfer(hidden_values[node], alpha)

    for output_node in range(num_output_nodes):
        output_transfer_derivative[output_node] = differentiate_transfer(output_values[output_node], alpha)
        error_times_transfer_deriv[output_node] = output_errors[output_node] * output_transfer_derivative[output_node]

    for hidden_node in range(num_hidden_nodes):
        weighted_output_errors[hidden_node] = 0
        for output_node in range(num_output_nodes):
            weighted_output_errors[hidden_node] += v_weights[output_node, hidden_node] * \
                                                   error_times_transfer_deriv[output_node]

    delta_hidden_biases = np.zeros(num_hidden_nodes)
    new_hidden_biases = np.zeros(num_hidden_nodes)

    for node in range(num_hidden_nodes):  # Number of rows in input-to-hidden weightMatrix
        partial_sse_w_bias_hidden[node] = -hidden_transfer_derivative[node] * weighted_output_errors[node]
        delta_hidden_biases[node] = -eta * partial_sse_w_bias_hidden[node]
        new_hidden_biases[node] = hidden_biases[node] + delta_hidden_biases[node]

    return new_hidden_biases


########################################################################################################################
#
########################################################################################################################


def train_neural_net(alph=1, eta=0.1, max_iter=100, epsl=0.01, num_hidden_nodes=10, letter_size=81, use_variants=0,
                     noise_percent=0, num_noisy_nodes=0):
    """
    :param alph:  slope of transfer function
    :param eta:  learning rate
    :param max_iter:  max number of times to retrain network, attempting to reach convergence
    :param epsl:  max acceptable SSE
    :param num_hidden_nodes:  number of hidden nodes in hidden layer
    :param letter_size:  size of letter "pixel grid"; 81 = 9x9, 25 = 5x5, etc.
    :param use_variants:  binary boolean; 1 = include letter variants in training data, 0 = only train with base letters
    :param noise_percent: percentage of pixels to become "noisy" (randomly assigned inverted pixels)
    :param num_noisy_nodes: number of input nodes to add noise to
    :return:  *** lots of cool stuff ***
    """

    num_train_sets, all_letters = load_letters(letter_size, use_variants)
    num_input_nodes = len(all_letters[0].array)
    num_output_nodes = 26  # possible letter classes

    all_letters = add_noise(all_letters, noise_percent, num_noisy_nodes)

    alpha = alph
    eta = eta
    max_num_iterations = max_iter
    epsilon = epsl
    iteration = 0

    # Initialize input-to-hidden and hidden-to-output weights with random values
    w_weights = initialize_weights((num_hidden_nodes, num_input_nodes))
    v_weights = initialize_weights((num_output_nodes, num_hidden_nodes))
    hidden_biases = initialize_bias_weights(num_hidden_nodes)
    output_biases = initialize_bias_weights(num_output_nodes)

    # Calc initial output values using randomly assigned weights and biases
    before_training = compute_training_outputs(alpha, num_train_sets, w_weights, hidden_biases, v_weights,
                                               output_biases, all_letters)

    while iteration < max_num_iterations:

        iteration += 1

        # Randomly select one letter for training
        training_letter_index = random.randint(0, num_train_sets-1)
        desired_outputs = np.zeros(num_output_nodes)  # initialize the output array with 0's
        desired_class = all_letters[training_letter_index].target  # identify the desired class
        desired_outputs[desired_class] = 1  # set the desired output for that class to 1

        # Pass training inputs through transfer function to hidden layer, and then pass hidden values through
        # transfer into output layer; calc errors (diff of predicted vs. actual)
        input_values = all_letters[training_letter_index].array
        hidden_values = feed_forward(alpha, input_values, w_weights, hidden_biases)
        output_values = feed_forward(alpha, hidden_values, v_weights, output_biases)
        output_errors = np.zeros(num_output_nodes)

        # Determine the error between actual and desired outputs
        new_sse = 0.0
        for node in range(num_output_nodes):  # Number of nodes in output set (classes)
            output_errors[node] = desired_outputs[node] - output_values[node]
            new_sse += output_errors[node] * output_errors[node]

        # Perform backpropagation of weight and bias changes
        new_v_weights = backprop_output_to_hidden(alpha, eta, output_errors, output_values, hidden_values, v_weights)
        new_output_biases = backprop_output_biases(alpha, eta, output_errors, output_values, output_biases)
        new_w_weights = backprop_hidden_to_input(alpha, eta, output_errors, output_values, hidden_values, input_values,
                                                 v_weights, w_weights)
        new_hidden_biases = backprop_hidden_biases(alpha, eta, output_errors, output_values, hidden_values, v_weights,
                                                   hidden_biases)

        # Replace old weights and biases, with new backpropagated versions
        v_weights = new_v_weights[:]
        output_biases = new_output_biases[:]
        w_weights = new_w_weights[:]
        hidden_biases = new_hidden_biases[:]

        # Feed forward new weights and biases back through model
        hidden_values = feed_forward(alpha, input_values, w_weights, hidden_biases)
        output_values = feed_forward(alpha, hidden_values, v_weights, output_biases)

        # Determine the new errors between actual and desired outputs
        new_sse = 0.0
        for node in range(num_output_nodes):  # Number of nodes in output set (classes)
            output_errors[node] = desired_outputs[node] - output_values[node]
            new_sse = new_sse + output_errors[node] * output_errors[node]

        if new_sse < epsilon:
            break

    # After training, get a new comparative set of outputs, errors, and sse
    after_training = compute_training_outputs(alpha, num_train_sets, w_weights, hidden_biases, v_weights,
                                              output_biases, all_letters)

    return {'before_training': before_training, 'after_training': after_training, 'convergence_iteration': iteration}


########################################################################################################################
#   FILE MANAGEMENT FUNCTIONS
########################################################################################################################

def write_results_to_file(results, parameters, base_dir='/Users/joecipolla/Documents/Reference/Education/'
                                                        '490-DL_Deep Learning/Week_5/param_testing3/'):
    """
    :param results: training results to write down
    :param parameters: sets filename = hyper-parameter settings
    :param base_dir: directory to write to
    :return:
    """
    results_file = open(str(base_dir + parameters + '.txt'), 'w')

    headers = str("letter | ")
    for key in results[parameters]['after_training']['A']:
        headers += str(key) + " | "
    headers = headers[:-2]  # trim off extra comma-space at end of string
    results_file.write("%s\n" % headers)

    for letter in results[parameters]['after_training']:
        values = str(letter) + " | "
        for value in results[parameters]['after_training'][letter].itervalues():
            value = np.array(value)
            value = np.array_repr(value).replace('\n', '').replace(' ', '').replace(',', ', ').\
                replace('array(', '').replace(')', '').replace('float64(', '')
            values += str(value) + " | "
        values = values[:-3]
        results_file.write("%s\n" % values)

    results_file.close()


def read_training_results_from_dir(base_dir='/Users/joecipolla/Documents/Reference/Education/'
                                            '490-DL_Deep Learning/Week_5/param_testing/'):
    """
    Opens training results files in base directory,
    and returns the selected file's "column" as a dictionary (filename as dict key, column values as dict values)
    (ie: letter, desired_output, output, hidden, errors, or new_sse)
    :return: results' column dictionary
    """
    from os import listdir

    result_files = [f for f in listdir(base_dir) if f[0] != '.']
    results_from_dir = {}
    for r_file in range(len(result_files)):
        with open(base_dir + result_files[r_file]) as f:
            desired_output = {}
            output = {}
            hidden = {}
            errors = {}
            new_sse = {}
            i = 0
            for line in f:
                if i == 0:
                    pass
                else:
                    values = line.split(' | ')
                    key = line[0]
                    desired_output[key] = values[1]
                    output[key] = values[2]
                    hidden[key] = values[3]
                    errors[key] = values[4]
                    new_sse[key] = values[5]
                i += 1
            results = {'desired_output': desired_output, 'output': output, 'hidden': hidden,
                       'errors': errors, 'new_sse': new_sse}
            results_from_dir[str(result_files[r_file]).replace('.txt', '')] = results

    return results_from_dir


def plot_results(results):
    alphas = []
    etas = []
    iterations_in_each_result = []
    for result in results:
        iterations_in_each_result.append(result["iterations"])
        alphas.append(result["alpha"])
        etas.append(result["eta"])

    plt.scatter(alphas, etas, s=iterations_in_each_result)
    plt.title("Iterations to Convergence")
    plt.xlabel('Sigmoid Transfer (alpha)')
    plt.ylabel('Learning Rate (eta)')
    plt.show()


########################################################################################################################
# MAIN module
########################################################################################################################

def main(run_network=1):

    welcome()

    # load hyperparameter results
    training_results = read_training_results_from_dir('/Users/joecipolla/Documents/Reference/Education/'
                                                      '490-DL_Deep Learning/Week_5/param_testing3/')

    # find parameter setting test with lowest average sse
    hyper_parameters = {}
    for test in training_results:
        total_sse = 0
        for key in training_results[test]['new_sse']:
            total_sse += float(training_results[test]['new_sse'][key])
        hyper_parameters[test] = total_sse / float(26)  # calc average sse

    lowest_avg_sse = min(hyper_parameters, key=hyper_parameters.get).split(', ')
    best_alpha = lowest_avg_sse[0].split('_')[1]
    best_eta = lowest_avg_sse[1].split('_')[1]
    best_num_of_hid = lowest_avg_sse[2].split('_')[2]

    print "best alpha: " + best_alpha
    print "best eta: " + best_eta
    print "best num of hidden nodes: " + best_num_of_hid
    print ''


    # fill dictionaries for each parameter's SSE by letter and overall
    alphas = {}
    etas = {}
    numHid = {}
    noisePerc = {}
    numNoisy = {}
    test_index = 0
    for test in training_results:
        # setup empty dicts for each letter in test cases and overall
        if test_index == 0:
            for key in training_results[test]['new_sse']:
                alphas[key + '_alpha'] = []
                alphas[key + '_sse'] = []
                etas[key + '_eta'] = []
                etas[key + '_sse'] = []
                numHid[key + '_numHid'] = []
                numHid[key + '_sse'] = []
                noisePerc[key + '_noisePerc'] = []
                noisePerc[key + '_sse'] = []
                numNoisy[key + '_numNoisy'] = []
                numNoisy[key + '_sse'] = []
                noisePerc['all_noisePerc'] = []
                noisePerc['avg_sse'] = []
                numNoisy['all_numNoisy'] = []
                numNoisy['avg_sse'] = []

        test_parameters = test.split(', ')
        noisePerc['all_noisePerc'].append(float(test_parameters[3].split('_')[2]))
        numNoisy['all_numNoisy'].append(float(test_parameters[4].split('_')[2]))

        noise_sse = 0
        # fill in parameter and sse values for each letter in each dictionary
        for key in training_results[test]['new_sse']:
            alphas[key + '_alpha'].append(float(test_parameters[0].split('_')[1]))
            alphas[key + '_sse'].append(float(training_results[test]['new_sse'][key]))
            etas[key + '_eta'].append(float(test_parameters[1].split('_')[1]))
            etas[key + '_sse'].append(float(training_results[test]['new_sse'][key]))
            numHid[key + '_numHid'].append(float(test_parameters[2].split('_')[2]))
            numHid[key + '_sse'].append(float(training_results[test]['new_sse'][key]))
            noisePerc[key + '_noisePerc'].append(float(test_parameters[3].split('_')[2]))
            noisePerc[key + '_sse'].append(float(training_results[test]['new_sse'][key]))
            numNoisy[key + '_numNoisy'].append(float(test_parameters[4].split('_')[2]))
            numNoisy[key + '_sse'].append(float(training_results[test]['new_sse'][key]))
            noise_sse += float(training_results[test]['new_sse'][key])
        noise_sse = noise_sse / len(noisePerc['A_sse'])
        noisePerc['avg_sse'].append(noise_sse)
        numNoisy['avg_sse'].append(noise_sse)

        test_index += 1


    # plot average SSE by alpha and eta
    y = noisePerc['avg_sse']
    x = noisePerc['all_noisePerc']
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 0.35, 0.05))
    ax.patch.set_facecolor('lightgrey')
    plt.scatter(x, y)
    plt.grid()
    plt.title('Average SSE vs. Noisy Pixels')
    plt.xlabel('Percent of Noisy Pixels')
    plt.ylabel('SSE')
    plt.xlim([-0.1, 1.1])
    plt.savefig('/Users/joecipolla/Documents/Reference/Education/490-DL_Deep Learning/Week_5/'
                'sse_by_noise_percent.png')
    plt.close()

    y = numNoisy['avg_sse']
    x = numNoisy['all_numNoisy']
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 90, 10))
    ax.set_yticks(np.arange(0, 0.35, 0.05))
    ax.patch.set_facecolor('lightgrey')
    plt.scatter(x, y, c='r')
    plt.grid()
    plt.title('Average SSE vs. Noisy Nodes')
    plt.xlabel('Number of Noisy Nodes')
    plt.ylabel('SSE')
    plt.xlim([-5, 85])
    plt.savefig('/Users/joecipolla/Documents/Reference/Education/490-DL_Deep Learning/Week_5/'
                'sse_by_num_noisy_nodes.png')
    plt.close()


    # test various hyperparameters
    if run_network == 1:
        alphas_to_try = [float(best_alpha)]  # np.arange(0, 2.5, 0.5)
        etas_to_try = [float(best_eta)]  # np.arange(0.5, 2.5, 0.5)
        num_of_hidden_nodes_to_try = [int(best_num_of_hid)]  # [6, 10, 16, 28, 36, 44]
        noise_percent_to_try = [0, 0.1, 0.25, 0.5, 0.75, 0.95]
        num_noisy_nodes_to_try = [0, 10, 20, 30, 40, 60, 81]

        training_results = {}
        for a in alphas_to_try:
            for e in etas_to_try:
                for hid_cnt in num_of_hidden_nodes_to_try:
                    for noise_percent in noise_percent_to_try:
                        for num_noisy_nodes in num_noisy_nodes_to_try:
                            parameter_settings = str('alpha_' + str(a) + ', eta_' + str(e) + ', num_hid_' +
                                                     str(hid_cnt) + ', noise_percent_' + str(noise_percent) +
                                                     ', noisy_nodes_' + str(num_noisy_nodes))
                            print "sys time: " + str(dt.datetime.now().time())
                            print parameter_settings
                            print ""
                            training_results[parameter_settings] = train_neural_net(a, e, 10000, 0.005, hid_cnt, 81, 0,
                                                                                    noise_percent, num_noisy_nodes)
                            write_results_to_file(training_results, parameter_settings)

    # observe change in SSE as different increments of noise are added to the input vectors

    # observe difference in SSE from base letters and variants

    print "thank you for playing. goodbye."

########################################################################################################################
# Conclude MAIN procedure
########################################################################################################################


if __name__ == "__main__":
    main(0)


########################################################################################################################
########################################################################################################################
# End program
