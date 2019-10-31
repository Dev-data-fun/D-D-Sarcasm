import torch
import torch.nn as nn
import fastai.layers as L
import torch.nn.functional as F
from collections import OrderedDict

torch.backends.cudnn.deterministic = True

def create_embedding_layer(weights_matrix, non_trainable=True):

    """Creates the embedding layer and loads the pre-trained embeddings from weights_matrix
       non-trainable = True : static embeddings
       non-trainable = False : non-static embeddings
       """

    num_embeddings, embedding_dim = weights_matrix.shape

    weights_matrix = torch.from_numpy(weights_matrix)   # Creates tensor from numpy array 

    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=1).to('cuda')
    # Padding index is kept zero and no gradients tracked
    emb_layer.load_state_dict({'weight': weights_matrix})
    # Loads pretrained vector weights

    if non_trainable:
        emb_layer.weight.requires_grad = False

    else:
        emb_layer.weight.require_grad = True

    return emb_layer

def convBlock(inplanes, self_attention = False, leaky = None):

    return nn.Sequential(L.conv_layer(inplanes, inplanes*2, padding=1, is_1d=True,
                                      self_attention=self_attention, leaky=leaky),
                         L.conv_layer(inplanes*2, inplanes*2, stride=2, padding=1, is_1d=True,
                                      self_attention=self_attention, leaky=leaky))


class ConvNet(nn.Module):

    def __init__(self, weights_matrix, inplanes = 16, num_blocks = 4,
                 self_attention = False, leaky=0.02, embed_dim = 50):

        super(ConvNet, self).__init__()

        self.embedding = create_embedding_layer(weights_matrix, False)

        self.init_conv = L.conv1d(1, inplanes, ks=(3, embed_dim), stride=1, padding=(1,0))

        self.features = nn.Sequential(OrderedDict([
            ('init_conv', self.init_conv),
            ('init_norm', nn.BatchNorm1d(inplanes)),
            ('init_relu', nn.ReLU(inplace=True))]))

        channels = inplanes

        for i in range(num_blocks):
            block = convBlock(channels, self_attention = self_attention, leaky=leaky)
            self.features.add_module('convblock%d' % (i+1), block)
            channels = channels*2
        
        self.relu = L.relu(inplace=True, leaky=leaky)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels*8)
        self.fc2 = nn.Linear(channels*8, channels*4)
        self.fc3 = nn.Linear(channels*4, channels*2)
        self.classifier = nn.Linear(2*channels, 2)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        features = self.embedding(x).unsqueeze(1)
        for i, layer in enumerate(self.features):

            if i == 1:
                features = features.squeeze(3)
            features = layer(features)

        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.classifier(out)
        
        return out
