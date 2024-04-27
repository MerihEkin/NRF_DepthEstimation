import torch
import torch.nn as nn

class SplitNodeLoss(nn.Module):
    def __init__(self):
        super(SplitNodeLoss, self).__init__()

    def forward(self, prob_left, loss_left, prob_right, loss_right):
        loss = torch.sum(torch.mul(prob_left, loss_left) + torch.mul(prob_right, loss_right))
        return loss
    
if __name__ == '__main__':
    # test
    prob_left = torch.tensor([0.1])  # Example probability tensor for the left node
    loss_left = torch.tensor([1.2])  # Loss associated with the left node
    prob_right = torch.tensor([0.2])  # Probability tensor for the right node
    loss_right = torch.tensor([1.3])  # Loss for the right node

    loss_function = SplitNodeLoss()
    loss = loss_function(prob_left, loss_left, prob_right, loss_right)
    print(f"Calculated Loss: {loss.item()}")
