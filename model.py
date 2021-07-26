import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.softmax = nn.Softmax()
        
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
    
    def forward(self, features, captions):
        embeds = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        embeds = torch.cat((features, embeds), 1)
        x, _ = self.lstm(embeds)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []

        for i in range(max_len) :
            x, states = self.lstm(inputs, states)
            x = self.fc(x.squeeze(1))
            x_max = x.max(1)[1]
            predicted = x_max.item()
            output.append(predicted)
            inputs = self.embed(x_max).unsqueeze(1)

        return output
