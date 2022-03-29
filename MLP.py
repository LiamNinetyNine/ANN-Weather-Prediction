# ----------------------------------------------------------------------------------------------------------------------

from this import d
from turtle import dot, forward
import numpy as np
import pandas as pd
from math import ceil, floor
import math
import random
import matplotlib.pyplot as plt
from time import perf_counter, sleep

# ----------------------------------------------------------------------------------------------------------------------

class Node():
    def __init__(self, index, is_output):
        self.weights = [] 
        self.is_output = is_output
        self.index_in_layer = index
        self.delta = None # Node error
        self.S = None # Node output (weighted sum not activated)
        self.next_nodes = [] # The nodes of the "next" layer i.e. the i + 1 layer
        self.new_weights = [] # stores the updated weights after deltas are calculated
        self.last_inputs = [] # Stores the last inputs into the node for use in updating the weights


    def set_next_nodes(self, next_layer):
        """Stores the nodes of the next layer in order to perform backpropagation (used when computing delta)"""
        self.next_nodes = []
        for next_node in next_layer:
            self.next_nodes.append(next_node)

    def sigmoid(self,z):
        """Activation function sigmoid"""
        return ( 1 / (1 + math.exp(-z)) )

    def sigmoid_prime(self, z):
        """The first derivative of the activation function sigmoid"""
        return ((self.sigmoid(z)) * (1 - self.sigmoid(z)))

    def tanh(self, z):
        """tanh activation function"""
        return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))

    def tanh_prime(self, z):
        """derivative of the tanh activation function"""
        return 1 - (self.tanh(z) ** 2)

    def predict(self, input, node_identifier_for_debug, initial):
        """Computes forward propagation of data through this node by computing the dot product"""
        self.last_inputs = []
        dot_prod = 0
        for weight, input_val in zip(self.weights, input):
            dot_prod += (weight * input_val)
            self.last_inputs.append(input_val) # Not sure about this
        self.S = dot_prod
        #print(f"Forward propagated {self.weights} and {input} for node {node_identifier_for_debug.index_in_layer} which gave the weighted sum {self.S} and activated sum {self.sigmoid(self.S)}")
        return self.sigmoid(dot_prod) #return u

    def compute_delta(self, y, node_identifier_for_debug): #delta is the error, y is the target value (what the output should've been)
        """Compute the delta for a node, used in backpropagation in order to adjust weights"""
        if self.is_output:
            f_prime_S = self.sigmoid_prime(self.S)
            self.delta = (y - self.sigmoid(self.S)) * f_prime_S
        else:
            delta_sum = 0
            for next_node in self.next_nodes:
                delta_sum += next_node.weights[self.index_in_layer] * next_node.delta
            self.delta = delta_sum * self.sigmoid_prime(self.S)
        #print(f"The delta value for node {node_identifier_for_debug.index_in_layer} is {self.delta} and the weighted sum for this node is {self.S}")

    def generate_weights(self, num_inputs_into_node):
        """Randomly generate n weights between [0,1] where n is the number of inputs into the node (+1 for the bias node)"""
        mu = 0
        sigma = 1 / math.sqrt(num_inputs_into_node)
        for _ in range(num_inputs_into_node + 1): #+1 for bias
            #self.weights.append(random.gauss(mu, sigma))
            #self.weights.append(0)
            self.weights.append(random.uniform(0,1))

    def update_weights(self, learning_rate):
        """Update the weights of the node after the delta (error) has been calculated"""
        # The gradient of the error 𝜕𝐸/𝜕𝑤 = (delta_j * u_i) (j is the row num)
        self.new_weights = [] # Make a new list to store the new weights
        for weight, last_input in zip(self.weights, self.last_inputs):
            #Since we computed obtained_value - target values in delta, we add the weights, not subtract them
            new_weight = weight + learning_rate * (self.delta * last_input) #Compute new weight
            self.new_weights.append(new_weight)
            #print(f"Updating the weights for node {self.index_in_layer} to {new_weight} from previous weight {weight} and the last input u_i has been recorded as {last_input}")
        self.weights = [] # Empty the previous weights to avoid any unnecessary errors
        for new_w in self.new_weights:
            self.weights.append(new_w) # Replace the old weights with the new weights

# ----------------------------------------------------------------------------------------------------------------------

class Layer():
    def __init__(self, num_nodes, is_output):
        self.nodes = []
        self.deltas = []
        self.is_output = is_output
        for i in range(num_nodes):
            node = Node(i, self.is_output)
            self.nodes.append(node) # Fills the layer with num_nodes amount of nodes

    def connect_layer(self, layer):
        """Store a layer of nodes in a previous layer"""
        for node in self.nodes:
            node.set_next_nodes(layer.nodes)
    
    def num_nodes(self):
        """returns the number of nodes in a layer"""
        return len(self.nodes)
    
    def init_layer(self, num_nodes_in_previous_layer):
        """Initialises the layer by generating the weights for each node"""
        for node in self.nodes:
            node.generate_weights(num_nodes_in_previous_layer)

    def forward_propagation(self, inputs, initial = False):
        """Takes an input row and forward propagates it through the layer by making predictions with each of its nodes"""
        inputs = list(inputs)
        inputs.append(1) #add a bias to the input
        predictions = []
        for node in self.nodes:
            predictions.append(node.predict(inputs, node, initial))
        return predictions

    def compute_deltas(self, y):
        """Computes the deltas of each nodes in the layer, used for backwards propagation"""
        for node in self.nodes:
            node.compute_delta(y, node)
    
    def update_weights_of_nodes(self, learning_rate):
        """Once the deltas have been computed, run this function to update the weights of each nodes in this layer"""
        for node in self.nodes:
            node.update_weights(learning_rate)

# ----------------------------------------------------------------------------------------------------------------------

class Network():
    def __init__(self, epochs, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = []
        
    def add_layer(self, num_nodes, is_output = False):
        """Adds a layer to the neural network structure"""
        if(is_output):
            output_layer = Layer(num_nodes, is_output)
            self.layers.append(output_layer)
        else:
            hidden_layer = Layer(num_nodes, is_output)
            hidden_layer.connect_layer(self.layers[0])
            self.layers.insert(0, hidden_layer)

    def train(self, X_train, y_train):
        """MLP learning algorithm from the lecture notes; trains the network on data by making predictions, 
        calculating the error and then updates the weights upwards or downwards with that error value in order
        to reach an optimum point where the loss (the incorrectness of the predictions) is as low as possible"""
        # Start timer for algorithm improvement metrics
        start = perf_counter() # Timer used to output how long it took to train the network
        # (0) INITIALIZE THE WEIGHTS AND BIASES
        self.layers[0].init_layer(num_nodes_in_previous_layer = len(X_train[0]))
        for i in range(1,len(self.layers)):
            self.layers[i].init_layer(num_nodes_in_previous_layer = self.layers[i-1].num_nodes())
        mean_squared_error_per_epoch = []
        debug = 0
        # REPEAT UNTIL X
        for _ in range(self.epochs):
            squared_error = []
            # (1) TAKE THE NEXT TRAINING EXAMPLE row AND THE NEXT CORRECT OUTPUT y
            for row, y in zip(X_train,y_train):
                #print(f"\nNew Row\n{row} with expected output {y}\n ")
                row = list(row)
                #if debug == 3:
                    #quit()
                # (2) MAKE A FORWARD PASS THROUGH THE NETWORK COMPUTING S,u FOR EVERY NODE IN THE LAYER
                y_predicted = self.predict(row)
                squared_error.append((y - y_predicted) ** 2)
                # (3) BACKWARDS PASS THROUGH THE NETWORK COMPUTING FOR EACH NODE IN EACH LAYER delta (the gradient of the activation fn)
                for layer in reversed(self.layers):
                    layer.compute_deltas(y)
                # (3) UPDATE THE WEIGHTS
                for layer in self.layers:
                    layer.update_weights_of_nodes(self.learning_rate)
                debug += 1
            mean_squared_error_per_epoch.append(float(sum(squared_error)/self.epochs)) # take the mean squared error of each prediction made on the dataset for each epoch
        end = perf_counter() # End the timer once all epochs are complete
        print(f"Time taken to train = {round(end-start,2)} seconds")
        # Plot the loss
        plt.plot(list(range(0,self.epochs)),mean_squared_error_per_epoch)
        plt.xlabel("Epoch number")
        plt.ylabel("MSE")
        plt.title("Loss per Epoch")
        plt.show()

    def predict(self, row):
        """Use the network to make a prediction once it's been trained by forward propagating new data row through the layers"""
        predicted_vals = self.layers[0].forward_propagation(row) # Forward propagate the inputted data
        for i in range(1, len(self.layers)):
            predicted_vals = self.layers[i].forward_propagation(predicted_vals) # Forward propagate the predicted, activated, values
        #print(f"Forward propagated and got {activations}")
        return predicted_vals # Return the last activated output from the final output layer

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read in the data
    df = pd.read_excel("C:\\Users\\liamm\\Desktop\\Coursework\\AI-Methods-ANN-Coursework\\Ouse.xlsx")

    # Organise inputs and output
    inputs = ["Crakehill S", "Skip Bridge S", "Westwick S"]
    X = df[inputs].to_numpy()
    output = 'Skelton S'
    y = df[output].to_numpy()

    # Split the data
    train_percent = 0.7 #70% of the training data
    train_split_X = ceil(len(X)*train_percent) #number of rows in train set for X
    train_split_y = ceil(len(y)*train_percent)

    print(f"Data successfully split with {train_percent*100}% train and {100-train_percent*100}% test")

    X_train = list(X[0:train_split_X+1:1]) #0 is starting index (inclusive), train_split is stopping index (exclusive so +1), 1 is step size
    X_test = list(X[train_split_X+1:len(X)+1:1])
    y_train = list(y[0:train_split_y+1:1])
    y_test = list(y[train_split_y+1:len(y)+1:1])

    # Create the Neural network
    network = Network(learning_rate = 0.7, epochs = 11)
    network.add_layer(num_nodes = 1, is_output = True)
    network.add_layer(num_nodes = 6)

    # Train the Network
    network.train(X_train, y_train)

    print("\n Training complete. \n Starting testing... \n")

    # Evaluate the model by using loss functions and graphs
    errors = []
    predictions = []
    for expected_result, test_row in zip(y_test, X_test):
        prediction = network.predict(row = test_row)
        predictions.append(prediction)
        errors.append((expected_result - prediction) ** 2)
    print(f"Loss = {sum(errors)/len(errors)}")

    # Sort data for graphing
    crakehill_X_test = []
    skip_bridge_X_test = []
    westwick_X_test = []
    for feature in X_test:
        crakehill_X_test.append(feature[0])
        skip_bridge_X_test.append(feature[1])
        westwick_X_test.append(feature[2])

    # Plot X_test features vs predictions
    plt.scatter(crakehill_X_test, list(y_test), color = 'red')
    plt.scatter(crakehill_X_test, list(predictions), color = 'blue')
    plt.xlabel("Crakehill")
    plt.ylabel("Skelton")
    a = np.array(y_test)
    b = np.array(predictions)
    mses = list(((a-b)**2).mean(axis=1))
    plt.title(f"MSE = {sum(mses)/len(mses)}")
    plt.show()

    plt.scatter(skip_bridge_X_test, list(y_test), color = 'red')
    plt.scatter(skip_bridge_X_test, list(predictions), color = 'blue')
    plt.xlabel("Skip Bridge")
    plt.ylabel("Skelton")
    a = np.array(y_test) # your x
    b = np.array(predictions) # your y
    mses = list(((a-b)**2).mean(axis=1))
    plt.title(f"MSE = {sum(mses)/len(mses)}")
    plt.show()

    plt.scatter(westwick_X_test, list(y_test), color = 'red')
    plt.scatter(westwick_X_test, list(predictions), color = 'blue')
    plt.xlabel("Westwick")
    plt.ylabel("Skelton")
    a = np.array(y_test) # your x
    b = np.array(predictions) # your y
    mses = list(((a-b)**2).mean(axis=1))
    plt.title(f"MSE = {sum(mses)/len(mses)}")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------