import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class CenterLoss(nn.Cell):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(Tensor(np.random.randn(self.num_classes, self.feat_dim), mstype.float32).to("cuda"))
        else:
            self.centers = nn.Parameter(Tensor(np.random.randn(self.num_classes, self.feat_dim), mstype.float32))

    def construct(self, x, labels):
        x = x.astype(ops.Float32())
        batch_size = x.shape[0]

        distmat = ops.Pow()(x, 2).sum(axis=1, keepdims=True).expand_dims(1).broadcast_to((batch_size, self.num_classes)) + \
                  ops.Pow()(self.centers, 2).sum(axis=1, keepdims=True).broadcast_to((self.num_classes, batch_size)).transpose()
        
        classes = ops.Arrange(0, self.num_classes, dtype=ops.Int64())
        if self.use_gpu:
            classes = classes.to("cuda")

        labels = labels.reshape(batch_size, 1).broadcast_to((batch_size, self.num_classes))
        mask = labels == classes

        dist = distmat * mask.astype(ops.Float32())
        loss = ops.ClipByValue()(dist, 1e-12, 1e+12).sum() / batch_size

        return loss
