from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import time
import random
import Queue


def PrintNetwork(network):
  
  print "Input 1 Weight 1: ", network.inputs[0].forward_weights[0].value 
  if len(network.hidden_nodes) > 0:
    print "Hidden 1 Forward Weight 1: ", network.hidden_nodes[0].forward_weights[0].value


# <--- Problem 3, Question 1 --->

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  network.CheckComplete()
  # 1) Assign input values to input nodes
  # 2) Propagates to hidden layer
  # 3) Propagates to the output layer

  # assign input values to input nodes
  for i, input_node in enumerate(network.inputs):
    input_node.transformed_value = input[i]

  # assign values to hidden layer
  for hidden_node in network.hidden_nodes:
    hidden_node.raw_value = network.ComputeRawValue(hidden_node)
    hidden_node.transformed_value = network.Sigmoid(hidden_node.raw_value)

  # finish output layer
  for output_node in network.outputs:
    output_node.raw_value = network.ComputeRawValue(output_node)
    output_node.transformed_value = network.Sigmoid(output_node.raw_value)

#< --- Problem 3, Question 2

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()
  # 1) We first propagate the input through the network
  # 2) Then we compute the errors and update the weigths starting with the last layer
  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  
  FeedForward(network, input)

  # output layer
  for m, output_node in enumerate(network.outputs):     
    a_m = output_node.transformed_value
    output_node.delta = (target[m] - a_m)*a_m*(1.0-a_m)

  # hidden layer first, then input layer 
  for node in reversed(network.hidden_nodes):
    error = 0.0
    for j in range(len(node.forward_neighbors)):
      error += node.forward_weights[j].value*node.forward_neighbors[j].delta

    a_m = node.transformed_value
    node.delta = error*a_m*(1-a_m) 

    # change forward weights
    for j in range(len(node.forward_weights)):
      node.forward_weights[j].value += learning_rate*node.transformed_value*node.forward_neighbors[j].delta

  # same code again for inputs
  for node in network.inputs:
    error = 0.0
    for j in range(len(node.forward_neighbors)):
      error += node.forward_weights[j].value*node.forward_neighbors[j].delta

    a_m = node.transformed_value
    node.delta = error*a_m*(1-a_m) 

    # change forward weights
    for j in range(len(node.forward_weights)):
      node.forward_weights[j].value += learning_rate*node.transformed_value*node.forward_neighbors[j].delta
    
      
# <--- Problem 3, Question 3 --->

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()

  for i in range(epochs):
    start = time.time()
    for j in range(len(inputs)):
      Backprop(network, inputs[j], targets[j], learning_rate)
    end = time.time()
    print "Time: ", end - start  
  

# <--- Problem 3, Question 4 --->

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initializatio.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() # < Don't remove this line >
    
  # <--- Fill in the methods below --->

  def EncodeLabel(self, label):
    """
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    # Replace line below by content of function
    encoding = [0.0]*10
    encoding[label] = 1.0
    return encoding

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    
    """
    # Replace line below by content of function
    encoding = map(lambda node: node.transformed_value, self.network.outputs)
    return encoding.index(max(encoding))

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    
    """

    # Replace line below by content of function
    pixels = [0.0]*(14*14)
    for i in range(len(image.pixels)):
      for j in range(len(image.pixels[i])):
        pixels[14*i+j] = image.pixels[i][j]/256.0

    return pixels      

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.
    
    """
    random.seed()
    for weights in self.network.weights:
      weights.value = random.uniform(-.01, .01)


#<--- Problem 3, Question 6 --->

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() # < Don't remove this line >
    
    # 1) Adds an input node for each pixel.    
    # 2) Add an output node for each possible digit label.
    print "Creating SimpleNetwork..."

    self.network = NeuralNetwork()

    # input nodes  
    for i in range(196):
      input_node = Node()
      self.network.AddNode(input_node, NeuralNetwork.INPUT)

    # add output nodes
    for i in range(10):
      node = Node()

      # link inputs to outputs
      for input_node in self.network.inputs:
        node.AddInput(input_node, None, self.network)

      self.network.AddNode(node, NeuralNetwork.OUTPUT)

    print "Done Creating SimpleNetwork!"
    self.network.CheckComplete()  


#<---- Problem 3, Question 7 --->

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel
    # 2) Adds the hidden layer
    # 3) Adds an output node for each possible digit label.
    print "Creating HiddenNetwork..."

    self.network = NeuralNetwork()

    #input nodes
    for i in range(196):
      input_node = Node()
      self.network.AddNode(input_node, NeuralNetwork.INPUT)

    #hidden nodes
    for i in range(number_of_hidden_nodes):
      hidden_node = Node()

      for input_node in self.network.inputs:
        hidden_node.AddInput(input_node, None, self.network)

      self.network.AddNode(hidden_node, NeuralNetwork.HIDDEN)

    #output nodes
    for i in range(10):
      output_node = Node()

      for hidden_node in self.network.hidden_nodes:
        output_node.AddInput(hidden_node, None, self.network)

      self.network.AddNode(output_node, NeuralNetwork.OUTPUT)

    print "Done Creating HiddenNetwork!"
    self.network.CheckComplete() 
    

#<--- Problem 3, Question 8 ---> 

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=10):
    """
    Arguments:
    ---------
    Your pick.

    Returns:
    --------
    Your pick

    Description:
    -----------
    Surprise me!
    """
    super(CustomNetwork, self).__init__() # <Don't remove this line>
    print "Creating CustomNetwork..."

    self.network = NeuralNetwork()

    #input nodes
    for i in range(196):
      input_node = Node()
      self.network.AddNode(input_node, NeuralNetwork.INPUT)

    #first layer hidden nodes
    first_hidden = []
    for i in range(number_of_hidden_nodes):
      hidden_node = Node()

      for input_node in self.network.inputs:
        hidden_node.AddInput(input_node, None, self.network)

      self.network.AddNode(hidden_node, NeuralNetwork.HIDDEN)
      first_hidden.append(hidden_node)

    #second layer hidden nodes
    second_hidden = []
    for i in range(number_of_hidden_nodes):
      hidden_node = Node()

      for input_node in first_hidden:
        hidden_node.AddInput(input_node, None, self.network)

      self.network.AddNode(hidden_node, NeuralNetwork.HIDDEN)
      second_hidden.append(hidden_node)

    #output nodes
    for i in range(10):
      output_node = Node()

      for hidden_node in second_hidden:
        output_node.AddInput(hidden_node, None, self.network)

      self.network.AddNode(output_node, NeuralNetwork.OUTPUT)

    print "Done Creating CustomNetwork!"
    self.network.CheckComplete() 


