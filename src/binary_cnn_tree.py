try:    # TODO : there has to be a better way to do this
    import node
    import cnn as ConvNet
except ImportError:
    try:
        import src.node as node
        import src.cnn as ConvNet
    except ImportError:
        raise ImportError("Cannot import node.")

import torch    

class BinaryTree:
    def __init__(self):
        self.depth = 10      # set tree depth fixed, value 10 comes from the paper
        cnn = ConvNet.ConvNet_TopOneThird()
        self.root = node.RootNode(cnn=cnn)
        self.leaf_nodes = []
        self.build_tree()

    def build_tree(self):
        self.add_children_to_node(root=self.root, level=2)

    def add_children_to_node(self, root : node.SplitNode, level):
        """
        Recursively add children nodes to root until traget tree depth is reached.
        """
        # print(f'Level : {level}')

        if level == self.depth:
            left_child = node.LeafNode(root=root)
            right_child = node.LeafNode(root=root)
            root.add_left_child(left_child)
            root.add_right_child(right_child)
            self.leaf_nodes.append(left_child)
            self.leaf_nodes.append(right_child)
            return
        
        if level < 4:
            cnn1 = ConvNet.ConvNet_TopOneThird()
            cnn2 = ConvNet.ConvNet_TopOneThird()
        elif level < 7:
            cnn1 = ConvNet.ConvNet_LowerOneThird()
            cnn2 = ConvNet.ConvNet_LowerOneThird()
        elif level < 10:
            cnn1 = ConvNet.ConvNet_BottomOneThird()
            cnn2 = ConvNet.ConvNet_BottomOneThird()
        
        left_child = node.SplitNode(root=root, cnn=cnn1)
        right_child = node.SplitNode(root=root, cnn=cnn2)

        root.add_left_child(left_child)
        root.add_right_child(right_child)

        self.add_children_to_node(root=left_child, level=level+1)
        self.add_children_to_node(root=right_child, level=level+1)

    def eval(self, x):
        return self.get_next_node(self.root, x)

    def get_next_node(self, root, x):
        if isinstance(root, node.LeafNode):
            return root.forward()
        elif isinstance(root, node.SplitNode):
            left_or_right = root.forward(x)
            if left_or_right == node.SplitNodeResult.left:
                return self.get_next_node(root=root.left_child, x=x)
            else:
                return self.get_next_node(root=root.right_child, x=x)

    def train_step(self):
        pass


if __name__ == '__main__':
    tree = BinaryTree()
    X = torch.randn(size=(3, 150, 150))    # for testing
    out = tree.eval(X)
    print(f'Depth estimate of the untrained binary tree is : {out}')



