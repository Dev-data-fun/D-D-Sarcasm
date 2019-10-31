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

class dropMergeLayer(nn.Module):
    """MergeLayer with dropout: Concatenates a modules input and output
            x.orig comes from fastai.layers.SequentialEx which is used in conjunction with this
            class"""
    def __init__(self, dropout = 0):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.dropout > 0: 
            x_drop = F.dropout(x, p=self.dropout, training=self.training)
            
        return torch.cat([x_drop, x.orig], dim=1) if self.dropout > 0 else torch.cat([x, x.orig], dim=1)

class _Transition(nn.Sequential):
    """Halves the dimension & number of features through 1x1 conv and 1D-AvgPool. 
        Used between denseBlocks"""
    def __init__(self, num_input_features, num_output_features, leaky = None):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('relu', L.relu(inplace=True, leaky = leaky))
        self.add_module('norm', nn.BatchNorm1d(num_output_features))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class _DenseBlock(nn.Sequential):
    """Creates a sequence  of <num_layers> denseblocks:
        num_input_features: Input size for initial denseblock
        growth_rate: Number of features each denseblock creates"""
    def __init__(self, num_layers, num_input_features, growth_rate, leaky = None, 
                 dropout = 0, self_attention = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = denseLayer(num_input_features + i * growth_rate, growth_rate, leaky = leaky,
                               dropout = dropout, self_attention = self_attention)
            self.add_module('denselayer%d' % (i + 1), layer)

def denseLayer(inplanes, growth = 32, leaky = None, dropout = 0, self_attention = False):
    
    return L.SequentialEx(L.conv_layer(inplanes, growth, padding=1, is_1d=True, leaky = leaky, self_attention=self_attention), dropMergeLayer(dropout = dropout))



class DweNet(nn.Module):
    """For building a densenet:
        block_config = Number of denselayers per respective denseBlock"""
    def __init__(self, weight_matrix, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64,  num_classes=2, leaky = 0.02, self_attention = False, 
                 dropout = 0.2, static = False, embed_dim=50):

        super(DweNet, self).__init__()

        self.embedding = create_embedding_layer(weight_matrix, static)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(1, num_init_features, kernel_size=(3, embed_dim), stride=1,
                                padding=(1,0), bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.LeakyReLU(leaky, True))
        ]))
        self.relu = L.relu(inplace=True, leaky=leaky)
        self.dropout = dropout
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                              growth_rate=growth_rate, leaky = leaky, dropout = self.dropout, 
                              self_attention = self_attention)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, leaky = leaky)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.fc1 = nn.Linear(num_features*2, num_features*8)
        self.fc2 = nn.Linear(num_features*8, num_features*4)
        self.classifier = nn.Linear(num_features*4, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        embedding = self.embedding(x).unsqueeze(1)
        features = embedding
        for i, layer in enumerate(self.features):
            if i == 1:
                #Needed after the initial conv to pass to batchnorm
                features = features.squeeze(3)
            features = layer(features)

        out = self.relu(features)
        out_avg = self.avgpool(out)
        out_max = self.maxpool(out)
        out = torch.cat([out_avg, out_max], 1).view(features.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.classifier(out)

        return out