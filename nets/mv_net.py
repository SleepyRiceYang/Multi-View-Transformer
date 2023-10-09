import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Tensor

class SiameseNetwork(nn.Cell):
    def __init__(self, num_views, latend_dim=256):
        super(SiameseNetwork, self).__init__()
        self.num_views = num_views
        self.latent_dim = latend_dim
        self.networks = nn.CellList([self._create_mobile_net_network() for _ in range(num_views)])
    
    def _create_mobile_net_network(self):
        mobilenetv2 = mobilenetv2_modified()
        mobilenetv2.conv1.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, pad_mode='valid', has_bias=False)
        in_features = mobilenetv2.classifier[1].out_channels
        mobilenetv2.classifier[1] = nn.Dense(in_features, self.latent_dim, weight_init='normal')
        return mobilenetv2
    
    def forward(self, x):
        assert x.shape[1] == self.num_views, 'Input tensor shape does not match num_views'
        latent_vectors = []
        for i in range(self.num_views):
            view_features = self.networks[i](x[:, i, :, :, :])
            latent_vectors.append(view_features)
        return ops.Stack()(latent_vectors, 1)

class AttentionPool(nn.Cell):
    
    def __init__(self, embedding_dim, num_views):
        super(AttentionPool, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_views = num_views

        self.attention_weights = nn.Parameter(Tensor(np.ones(num_views), mstype.float32))
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, input_data):
        attention_scores = self.attention_weights.view(1, -1)

        attention_weights = self.softmax(attention_scores)

        weighted_sum = ops.ReduceSum()(attention_weights.unsqueeze(-1) * input_data, axis=1)
        return weighted_sum

class MLP(nn.Cell):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_size = 512
        dropout = 0.1
        
        self.layers = nn.SequentialCell([
            nn.Dense(input_dim, output_dim, weight_init='normal'),
            nn.BatchNorm1d(output_dim)
        ])

    def construct(self, x):
        return self.layers(x)

class MLPHead(nn.Cell):
    def __init__(self, input_size, num_classes):
        super(MLPHead, self).__init__()
        hidden_size = 512
        feature_dim = 16
        self.fc1 = nn.Dense(input_size, hidden_size, weight_init='normal')
        self.relu = nn.ReLU()
        self.layers = nn.SequentialCell([
            nn.Dense(hidden_size, 256, weight_init='normal'),
            nn.Dropout(0.1)
        ])
        self.fc2 = nn.Dense(256, feature_dim, weight_init='normal')
        self.prelu_fc2 = nn.PReLU()
        self.classifier = nn.Dense(feature_dim, num_classes, weight_init='normal')

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.fc2(x)
        x = self.prelu_fc2(x)
        y = self.classifier(x)
        return x, y

class Net(nn.Cell):
    def __init__(self, cube_size=None, config=None, shape=None, num_classes=None):
        super().__init__()

        self.views = 9
        self.latend_dim = 768
        self.embedding_dim = 768
        self.num_classes = 2
        self.feat_dim = 16

        self.siamese_net = SiameseNetwork(num_views=self.views, latend_dim=768)
        self.attention_pool = TransformerEncoder(num_layers=1, embedding_dim=768, num_heads=8)
        self.MLPHead = MLPHead(input_size=self.embedding_dim * self.views, num_classes=2)
        self.MLPHeadBag = MLPHead(input_size=self.embedding_dim, num_classes=2)

        self.MLPHeadView = MLPHead(input_size=self.embedding_dim, num_classes=2)
        self.bn_views = nn.BatchNorm1d(self.views)

        self.bn_out = nn.BatchNorm1d(2)
        self.bn_feat = nn.BatchNorm1d(self.feat_dim)
       
    # construct  
    def construct(self, x):
        bc, c, d, h, w = x.shape
        x = get_multi_views(x, self.views)
        x = self.siamese_net(x)

        mode = 'attention'

        if mode == 'attention':
            before_norm_x = x
            x = self.bn_views(x)
            
            features_ins, logits_ins = self.MLPHeadView(x.view(bc*self.views, -1))
            features_ins = self.bn_feat(features_ins)
            logits_ins = self.bn_out(logits_ins)

            x = self.attention_pool(before_norm_x)
            
            x = ops.ReduceMean()(x, 1)
        
            features_bag, logits_bag = self.MLPHeadBag(x)
            
            features_bag = self.bn_feat(features_bag)
            logits_bag = self.bn_out(logits_bag)

            outputs_bag = nn.Softmax()(logits_bag, 1)
            outputs_ins = nn.Softmax()(logits_ins, 1)
            
        return outputs_ins, features_ins, outputs_bag, features_bag

class MultiheadSelfAttention(nn.Cell):
    def __init__(self, embedding_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.views = 9

        dropout = 0.1
        hidden_dim = 1024
        
        self.query = nn.Dense(embedding_dim, embedding_dim, weight_init='normal')
        self.key = nn.Dense(embedding_dim, embedding_dim, weight_init='normal')
        self.value = nn.Dense(embedding_dim, embedding_dim, weight_init='normal')
        
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.batch_norm1 = nn.BatchNorm1d(self.views)
        self.batch_norm2 = nn.BatchNorm1d(self.views)
        
        self.fc_out = nn.SequentialCell([
            nn.Dense(embedding_dim, hidden_dim, weight_init='normal'),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Dense(hidden_dim, embedding_dim, weight_init='normal'),
            nn.Dropout(dropout)
        ])

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def construct(self, x):
        batch_size, patches, embedding_dim = x.shape
        before_norm_x = x
        x = self.layer_norm1(x)

        queries = self.query(x).view(batch_size, patches, -1)
        keys = self.key(x).view(batch_size, patches, -1)
        values = self.value(x).view(batch_size, patches, -1)

        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        scores = ops.BatchMatMul()(queries, keys.transpose(0, 1, 3, 2)) / self.head_dim ** 0.5
        attention_weights = nn.Softmax(axis=-1)(scores)
        attention_output = ops.BatchMatMul()(attention_weights, values)

        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, patches, -1)
        
        x = before_norm_x + attention_output
        before_norm_x = x

        x = self.layer_norm2(x)

        output = self.fc_out(x) + before_norm_x
        return output

class TransformerEncoder(nn.Cell):

    def __init__(self, num_layers, embedding_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.SequentialCell([
            MultiheadSelfAttention(embedding_dim, num_heads) for _ in range(num_layers)
        ])

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def get_multi_views(data, num_views):
    bc, c, z, y, x = data.shape
    assert c == 1
    
    pos_idx = np.array([i % z for i in range(z)])
    neg_idx = np.array([-(i % z) + z - 1 for i in range(z)])
    
    output = []
    views = num_views

    for i in range(bc):
        plane1 = data[i, :, z // 2, :, :]
        plane2 = data[i, :, :, y // 2, :]
        plane3 = data[i, :, :, :, x // 2]
        
        plane4 = data[i, :, :, pos_idx, pos_idx]
        plane5 = data[i, :, :, pos_idx, neg_idx]
        
        plane6 = data[i, :, pos_idx, pos_idx, :]
        plane7 = data[i, :, pos_idx, neg_idx, :]
        
        plane8 = data[i, :, pos_idx, :, pos_idx].reshape(c, z, z)
        plane9 = data[i, :, pos_idx, :, neg_idx].reshape(c, z, z)
        
        if views == 9:
            planes = ops.Concat(0)((plane1, plane2, plane3, plane4, plane5, plane6, plane7, plane8, plane9))

            output.append(planes)

        elif views == 3:
            planes = ops.Concat(0)((plane1, plane2, plane3))

            output.append(planes)
    
    output = ops.Stack()(output, 0).reshape(bc, views, 1, z, z)
    return output
