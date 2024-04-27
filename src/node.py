import torch
import torch.nn as nn
import torch.optim as optim

from enum import Enum
from abc import ABC, abstractmethod

from random import gauss, uniform

try:    # TODO : there has to be a better way to do this
    from loss import SplitNodeLoss
except ImportError:
    try:
        from src.loss import SplitNodeLoss
    except ImportError:
        raise ImportError("Cannot import SplitNodeLoss")

DEBUG = True

class Mode(Enum):
    eval = 0
    train = 1

class SplitNodeResult(Enum):
    left = 0
    right = 1

class Node(ABC):
    """
    This implementation with random forests contain different nodes
    each type of node has common functions and attributes shared but 
    may implement it differently so we use a node base class. 
    """

    mode  = Mode.train  # class variable, required for switching between training and inference

    def __init__(self, root):
        super().__init__()
        """
        Every node has an output for split nodes this is the probability of going either left or right
        for leaf nodes it is the depth value

        Every node has a probability of reaching that node in the tree this value is very important for training

        Every node has a loss value that propagates backwards during training

        Every node has a root, except for tree root in that case root = None
        """
        self.output = 0.0
        self.loss = None
        self.root = root
        self.probability_of_reaching_node = None

    @abstractmethod
    def forward(self, x):
        "Every node must implement forward pass"
        pass

    @abstractmethod
    def train(self):
        "Every node must implement train"
        pass

    @classmethod
    def train_mode(cls):
        """Switching between train and evaluation modes are
        implemented with class methods so once changed it is applied
        to every node in the tree all at once
                
        it is only callable from Node, so that it is implemented for 
        Split and Leaf nodes at the same time"""
        if cls.__name__ != 'Node':
            raise TypeError("train_mode() can only be called from Node base class")
        cls.mode = Mode.train
        if hasattr(cls, 'model'):
            cls.model.train()

    @classmethod
    def eval_mode(cls):
        """Switching between train and evaluation modes are
        implemented with class methods so once changed it is applied
        to every node in the tree all at once
        
        it is only callable from Node, so that it is implemented for 
        Split and Leaf nodes at the same time"""
        if cls.__name__ != 'Node':
            raise TypeError("eval_mode() can only be called from Node base class")
        cls.mode = Mode.eval
        if hasattr(cls, 'model'):
            cls.model.eval()


class SplitNode(Node):
    def __init__(self, root, cnn : nn.Module, lr = 0.01):
        """
        Every split node has a Convolutional Neural Network Model with a sigmoid output
        this output serves as a probability of going to a child node. 
        
        let's fix this by saying this output is represent the probability of going to the
        left node. 

        each cnn needs to have its own optimizer (this is either necessary or will cause 
        inefficiency in training) 

        lr = learning rate
        """
        super().__init__(root)
        self.model = cnn
        self.left_child = None
        self.right_child = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = SplitNodeLoss()
        self.model.train()

    def add_left_child(self, node : Node):
        if self.left_child:
            raise AssertionError('Left child already exists.')
        self.left_child = node
        assert self.left_child.root == self    # for insurance

    def add_right_child(self, node : Node):
        if self.right_child:
            raise AssertionError('Right child already exists.')
        self.right_child = node
        assert self.right_child.root == self    # for insurance

    def forward(self, x):
        if self.probability_of_reaching_node is None:
            raise ValueError("Probability of reaching the node is not defined.")
        
        """
        First get model output, 
        x = image, 
        f(x) = probability of going left node
        """
        self.output = self.model(x)

        """
        Set probabilities of reaching left and right nodes by multiplying the probability 
        of reaching current node with the outputs
        """
        self.left_child.probability_of_reaching_node = self.probability_of_reaching_node * self.output
        self.right_child.probability_of_reaching_node = self.probability_of_reaching_node * (1 - self.output)

        "generate random value to for chosing right or left child node"
        random_value = uniform(0, 1)

        return SplitNodeResult.left if random_value < self.output else SplitNodeResult.right
    
    def train(self):
        if self.left_child.loss is None or self.right_child.loss is None:
            raise ValueError("Current loss depends on children node's loss.")
        self.model.train()
        self.optimizer.zero_grad()
        prob_right = 1.0 - self.output.item() 
        prob_right = torch.tensor([prob_right])
        self.loss = self.criterion(self.output, self.left_child.loss, prob_right, self.output)
        self.loss.backward()
        self.optimizer.step()


class LeafNode(Node):
    def __init__(self, root : Node):
        """
        The leaf nodes does not contain CNN 
        after a leaf node is reached the leaf node generates a depth estimate
        from its randomly initialized gaussian distribution 
        """
        super().__init__(root)
        self.mu = uniform(0, 5)     # TODO : check the paper again for generation of these values
        self.sigma = uniform(0, 1)  # TODO : maybe convert these to torch.random

    def forward(self):
        """
        Leaf Nodes output the depth estimate
        the depth estimate is sampled from a Gaussian Distribution
        mean and standard variance of the Gaussian Distribution are 
        class attributes
        """
        self.output = gauss(mu=self.mu, sigma=self.sigma)   # TODO : maybe convert these to torch.random
        return self.output 
    
    def train(self, depth):
        """
        TODO : check out the paper again for training lead nodes
        Does the distribution variables change or the inference weights
        of the forest ?? 
        """
        self.loss = torch.tensor([depth - self.output])
        return super().train()
    

class RootNode(SplitNode):
    def __init__(self, cnn):
        super().__init__(root=None, cnn=cnn)
        self.probability_of_reaching_node = 1.0


if __name__ == '__main__':  
    # For testing
    pass