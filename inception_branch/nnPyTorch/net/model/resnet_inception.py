#  download from:
#       https://raw.githubusercontent.com/pytorch/vision/master/torchvision/models/resnet.py
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.model.resnet_comb import resnet34
from net.model.inceptionv3_comb import Inception3

__all__ = ['ResNet_Inception']


class ResNet_Inception(nn.Module):

    def __init__(self, in_shape_inc=(1, 244, 244), in_shape_res=(3, 244, 244), num_classes=17):

        super(ResNet_Inception, self).__init__()
        self.model_inc = Inception3(in_shape_inc, num_classes=num_classes)
        self.model_res = resnet34(in_shape=in_shape_res, num_classes=num_classes)
        #2560 = 2048(inception feature) + 512(resnet feature)
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x_inc, x_res):
        x_inc = self.model_inc(x_inc)
        x_res = self.model_res(x_res)
        mixed_cat = torch.cat([x_inc, x_res], 1)
        x = self.fc(mixed_cat)

        logit = x
        prob  = F.sigmoid(logit)
        return logit, prob



##########################################################################
# if __name__ == '__main__':
#     print('%s: calling main function ... ' % os.path.basename(__file__))

#     # https://discuss.pytorch.org/t/print-autograd-graph/692/8
#     batch_size = 1
#     num_classes = 17
#     C, H, W = 3, 256, 256

#     inputs = torch.randn(batch_size, C, H, W)
#     labels = torch.randn(batch_size, num_classes)
#     in_shape = inputs.size()[1:]

#     if 1:
#         net = resnet34(in_shape=in_shape,
#                        num_classes=num_classes).cuda().train()

#         x = Variable(inputs)
#         logits, probs = net.forward(x.cuda())

#         loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels.cuda()))
#         loss.backward()

#         print(type(net))
#         print(net)

#         print('probs')
#         print(probs)

#         #input('Press ENTER to continue.')


##
#  max memory usage : resnet50:(96,3,224,224)   8501MiB
