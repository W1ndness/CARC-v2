from torch import nn
from torchsummary import summary


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, layers, dropout=False):
        super().__init__()
        assert input_dim == layers[0][0]
        assert num_classes == layers[-1][1]
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.net = nn.Sequential()
        for idx, layer in enumerate(layers[:-1]):
            self.net.add_module(f"Linear Layer {idx + 1}", nn.Linear(*layer))
            if dropout:
                self.net.add_module(f"Dropout Layer {idx + 1}", nn.Dropout())
            self.net.add_module(f"ReLU Layer", nn.ReLU())
        self.net.add_module(f"Linear Layer {len(layers)}",
                            nn.Linear(*layers[-1]))
        # self.net.add_module(f"Softmax Layer", nn.Softmax(num_classes))
        print("====== MLP Classifier Network Architecture ======")
        print(self.net)
        print("=================================================")

    def forward(self, X):
        return self.net(X)
