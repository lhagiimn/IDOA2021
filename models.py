import torch.nn as nn

class Net(nn.Module):

    def __init__(self, arch, out_dim):
        super(Net, self).__init__()
        self.arch = arch
        self.class_out = nn.Linear(1000, out_dim)


    def forward(self, inputs):

        cnn_features = self.arch(inputs)
        out = self.class_out(cnn_features)
        #reg_out = self.reg_out(cnn_features)

        return out







