from torch import nn

class SLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SLP, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    """
    For learning purposes I am to write a single layer perceptron using PyTorch.
    Not including any dataloading or anything else, just the model.
    """
    # Define the model
    model = SLP(2, 1)
    
    # Print the model summary
    print(model)