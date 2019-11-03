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

class dropsampleMergeLayer(nn.Module):
    """Mergelayer with dropout and downsampling
        used within the transition blocks"""
    def __init__(self, inplanes, dense: bool = False, dropout = 0):
        super().__init__()
        self.downsample = L.conv1d(inplanes, inplanes*2, stride=2)
        self.dropout = dropout

    def forward(self, x):
        if self.dropout > 0:
            x_drop = F.dropout(x, p=self.dropout, training=self.training)
        return (x_drop + self.downsample(x.orig)) if self.dropout > 0 else (x + self.downsample(x.orig))


def transition(inplanes, dropout = 0):

    return L.SequentialEx(
        L.conv1d(inplanes, inplanes*2, ks = 3, stride=2, padding = 1),
        dropsampleMergeLayer(inplanes, dropout = dropout)
    )

def resLayer(inplanes, self_attention = False, bottle = False, leaky=None):

    conv_kwargs = {'is_1d' : True, 'self_attention' : self_attention, 'leaky' : leaky}
    
    return L.res_block(inplanes, dense=False, bottle=bottle, **conv_kwargs)

class ResNet(nn.Module):

    def __init__(self, weights_matrix, layers = (2, 2, 5, 5), inplanes = 16, bottle=False, zero_init_residual=False, dropout = 0, leaky = None, embed_dim = 50, static = False):

        super(ResNet, self).__init__()

        self.embedding = create_embedding_layer(weights_matrix, static)
        self.init_conv = L.conv1d(1, inplanes, ks=(3, embed_dim), stride=1, 
                                  padding=(1,0), bias=False)
        
        self.dropout = dropout
        self.features = nn.Sequential(OrderedDict([
        ('init_conv', self.init_conv),
        ('init_norm', nn.BatchNorm1d(inplanes)),
        ('init_relu', nn.ReLU(inplace=True))]))
        
        num_features = inplanes
        for i, layer in enumerate(layers):
            self.features.add_module('resblock%d' %(i+1), 
                                     self._make_block(num_features, layer, leaky=leaky))
            if i!= len(layers)-1:
                self.features.add_module('transition%d' %(i+1), 
                                         transition(num_features, dropout = self.dropout))                         
                num_features = num_features*2
        self.relu = L.relu(inplace=True, leaky=leaky)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_features, num_features*2)
        self.classifier = nn.Linear(num_features*2, 2)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_block(self, outplanes, layer_count, leaky=None):

        layers = []

        for i in range(0, layer_count):
            layers.append(resLayer(outplanes, leaky=leaky))
            if self.dropout > 0 and i != layer_count-1:
                layers.append(nn.Dropout(p=self.dropout, inplace = True))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.embedding(x).unsqueeze(1)  
                  
        for i, layer in enumerate(self.features):
            if i == 1:
                x = x.squeeze(3)
            x = layer(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.classifier(x) 

        return x