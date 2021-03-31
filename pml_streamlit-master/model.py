import torch
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, class_num, args):
        super(CharacterLevelCNN, self).__init__()
        self.loss = args.get('Train', 'criterion')
        self.dropout_input = nn.Dropout2d(args.getfloat('Model', 'dropout_input'))
        self.feature_num = args.getint('Model', 'feature_num')
        self.conv1 = nn.Sequential(nn.Conv1d(args.getint('DataSet', 'char_num'),
                                             self.feature_num,
                                             kernel_size=7),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv2 = nn.Sequential(nn.Conv1d(self.feature_num, self.feature_num, kernel_size=7),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(self.feature_num, self.feature_num, kernel_size=3),
                                   nn.ReLU()
                                   )

        self.conv4 = nn.Sequential(nn.Conv1d(self.feature_num, self.feature_num, kernel_size=3),
                                   nn.ReLU()
                                   )

        self.conv5 = nn.Sequential(nn.Conv1d(self.feature_num, self.feature_num, kernel_size=3),
                                   nn.ReLU()
                                   )

        self.conv6 = nn.Sequential(nn.Conv1d(self.feature_num, self.feature_num, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        # compute the  output shape after forwarding an input to the conv layers
        # 128 is the batch size in the paper
        #input_shape = (args.getint('Train', 'batch_size'),
        #               args.getint('DataSet', 'l0'),
        #               args.getint('DataSet', 'char_num'))

        #input_shape = (args.getint('Train', 'batch_size'),
        #               args.getint('DataSet', 'char_num'),
        #               args.getint('DataSet', 'l0'))


        #self.output_dimension = self._get_conv_output(input_shape)

        # compute output shape after papers rule, still needs verification
        self.output_dimension = (int(((args.getint('DataSet', 'l0') - 96)/27) * self.feature_num))

        # define linear layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(1024, class_num)

        if self.loss == 'nllloss':
            self.log_softmax = nn.LogSoftmax(dim=1)
        # initialize weights

        self._create_weights()

    # utility private functions
    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        #x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    # forward
    def forward(self, x):
        #print(x.size())
        #x = x.transpose(1, 2)
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        if self.loss == 'nllloss':
            x = self.log_softmax(x)

        return x
