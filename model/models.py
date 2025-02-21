import torch

class SwishActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x * torch.sigmoid(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_input

class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, inputs: int, classes = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(inputs + 6 , 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.swish = SwishActivation()
        self.batchNorm = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, metadata = NotImplemented):
        x = self.fc1(x)
        x = self.batchNorm(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x = self.softmax(x)
        return x