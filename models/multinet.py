import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class SELayer(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias = False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias = False),
                nn.Sigmoid()
                )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        
        return out

class MeanProxy(nn.Module):
    def __init__(self, dim = 1):
        super(MeanProxy, self).__init__()
        self.dim = 1

    def forward(self, x):

        return torch.mean(x, dim = self.dim, keepdim = False)

class SumProxy(nn.Module):
    def __init__(self, dim = 1):
        super(SumProxy, self).__init__()
        self.dim = 1
    
    def forward(self, x):

        return torch.sum(x, dim = self.dim, keepdim = False)

class ConvClassifier(nn.Module):    
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv3d(2, 4, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(4),
                nn.ReLU(),
                nn.Conv3d(4, 1, kernel_size = 3, padding = 1),
                nn.BatchNorm3d(1),
                nn.ReLU(),
                )
        self.layer2 = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class CosineClassifier(nn.Module):
    def __init__(self):
        super(CosineClassifier, self).__init__()
        self.cosine = nn.CosineSimilarity()
    
    def forward(self, support, query):
        support = support.view(support.shape[0], -1)
        query = query.view(query.shape[0], -1)
        out = self.cosine(support, query)
        
        return out

class FCClassifier(nn.Module):
    def __init__(self, input_size = 64, hidden_size = 8, model_type = "ConvNet4"):
        super(FCClassifier, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(300 * 2, input_size, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(input_size, momentum=1, affine=True),
                nn.ReLU(),
                #nn.MaxPool2d(2)
                )

        self.layer2 = nn.Sequential(
                nn.Conv2d(input_size, 64,kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                #nn.MaxPool2d(2)
                )

        self.fc1 = nn.Linear(64 * 5 * 5, hidden_size)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,1)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out

class EuclideanClassifier(nn.Module):
    def __init__(self):
        super(EuclideanClassifier, self).__init__()

    def forward(self, support, query):

        support = support.reshape(support.shape[0], support.shape[1] * support.shape[2] * support.shape[3])
        query = query.reshape(query.shape[0], query.shape[1] * query.shape[2] * query.shape[3])
        #query = query.view(query.shape[0], -1)

        logits = torch.pow(support - query, 2).sum(dim = 1)

        return logits

class Proxy(nn.Module):
    def __init__(self, num_shot = 5, input_dim = 32, is_softmax = False):
        super(Proxy, self).__init__()
        self.is_softmax = is_softmax
        self.pooling = nn.AdaptiveAvgPool3d((input_dim, 5, 5))
        self.num_shot = num_shot
        self.layer = nn.Sequential(
                nn.Linear(input_dim * 2 * 5 * 5, 32, bias = False),
                nn.ReLU(),
                nn.Linear(32, 1, bias = False),
                nn.Sigmoid()
                )
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, x):
        out = None
        out_x = torch.sum(x, dim = 1)
        out_x = out_x.squeeze(1)
        out_x = self.pooling(out_x)
        out_x = out_x.view(out_x.shape[0], -1)
        for i in range(x.shape[0]):
            new_x = x[i, ...]
            tmp_x = x[i, ...]
            tmp_out = out_x[i, ...]
            tmp_out = tmp_out.repeat(tmp_x.shape[0], 1)
            tmp_x = self.pooling(tmp_x)
            tmp_x = tmp_x.view(tmp_x.shape[0], -1)
            tmp_x = torch.cat((tmp_x, tmp_out), dim = 1)
            tmp_x = self.layer(tmp_x)
            if self.is_softmax:
                tmp_x = self.softmax(tmp_x)
            tmp_x = tmp_x.squeeze(1)
            shape = new_x.shape
            new_x = torch.mm(tmp_x.unsqueeze(0), new_x.contiguous().view(new_x.shape[0], -1))
            new_x = new_x.reshape((1, 1, shape[-3], shape[-2], shape[-1]))
            if out is None:
                out = new_x
            else:
                out = torch.cat((out,new_x), dim = 0)

        return out.squeeze(1)

class MultiModalNet(nn.Module):
    def __init__(self, config):
        super(MultiModalNet, self).__init__()
        self.num_shot = config.num_shot
        self.num_way = config.num_way
        self.num_query = config.num_query
        self.model_type = config.model_type
        if config.model_type == "ConvNet4":
            from networks.convnet import ConvNet4
            self.encoder = ConvNet4(pooling = True)
            #self.semantic_encoder = ConvNet4(pooling = True)
            self.input_channels = 128
        elif config.model_type == 'AttentionConvNet4':
            from networks.convnet import AttentionConvNet4
            self.encoder = AttentionConvNet4(pooling = True)
            self.input_channels = 64
        elif config.model_type == 'ConvNet6':
            from networks.convnet import ConvNet6
            self.encoder = ConvNet6()
            self.input_channels = 64
        elif config.model_type == 'ResNet10':
            from networks.resnet import ResNet10
            self.encoder = ResNet10()
            self.input_channels = 512
        elif config.model_type == 'ResNet12':
            from networks.resnet12 import ResNet12
            self.encoder = ResNet12(keep_prob = 1)
            self.input_channels = 640
        elif config.model_type == "ResNet18":
            from networks.resnet import ResNet18
            self.encoder = ResNet18()
            self.input_channels = 512
        elif config.model_type == "ResNet34":
            from networks.resnet import ResNet34
            self.encoder = ResNet34()
            self.input_channels = 512
        elif config.model_type == "WRN28":
            from networks.wrn28 import Wide_ResNet
            self.encoder = Wide_ResNet(depth = 28, widen_factor = 10, dropout_rate = 0.8)
            self.input_channels = 640
        elif config.model_type == "VGG16":
            from networks.vgg import VGG16
            self.encoder = VGG16()
            self.input_channels = 640
        else:
            raise ValueError('')

        if config.proxy_type == "Sum":
            self.proxy = SumProxy(dim = 1)
        elif config.proxy_type == "Mean":
            self.proxy = MeanProxy(dim = 1)
        elif config.proxy_type == "Proxy" and config.classifier == "Euclidean": 
            self.proxy = Proxy(num_shot = self.num_shot, is_softmax = True)
        elif config.proxy_type == "Proxy" and config.classifier != "Euclidean":
            self.proxy = Proxy(num_shot = self.num_shot, is_softmax = False)
        else:
            raise ValueError("")

        if config.classifier == "3DConv":
            self.classifier = ConvClassifier()
        elif config.classifier == "FC":
            self.classifier = FCClassifier(self.input_channels, model_type = config.model_type)
        elif config.classifier == "Euclidean":
            self.classifier = EuclideanClassifier()
        elif config.classifier == "Cosine":
            self.classifier = CosineClassifier()
        else:
            raise ("Classifier value error")

        self.se = SELayer(self.input_channels)

        if config.teacher_model == "Bert":
            self.norm = nn.LayerNorm(768)
            self.semantic_fc = nn.Linear(self.input_channels, 768)
            self.key_fc = nn.Linear(self.input_channels, 768)
            self.value_fc = nn.Linear(self.input_channels, 768)
        else:
            self.norm = nn.LayerNorm(300)
            self.semantic_fc = nn.Linear(self.input_channels, 300)
            self.key_fc = nn.Linear(self.input_channels, 300)
            self.value_fc = nn.Linear(self.input_channels, 300)
            self.fc = nn.Linear(self.input_channels * 5 * 5, 4096)

        self.semantic_conv = nn.Conv2d(25, 1, kernel_size = 1)
        if config.model_type == "WRN28":
            self.semantic_conv = nn.Conv2d(21 * 21, 1, kernel_size = 1)
            #self.fc = nn.Linear(self.input_channels * 21 * 21, 4096)

        self.scale = config.scale
        #self.scale = 300 ** -0.5

    def forward(self, support_image, query_image):
        support = self.encoder(support_image)
        query = self.encoder(query_image)
        shape = support.shape
        
        if isinstance(self.classifier, ConvClassifier):
            support = self.se(support)
            query = self.se(query)

        support_feature = rearrange(support, "b c h w -> b (h w) c")
        query_feature = rearrange(query, "b c h w -> b (h w) c")
        support_feature = self.semantic_fc(support_feature)
        query_feature = self.semantic_fc(query_feature)

        visual_feature = torch.cat((self.semantic_conv(support_feature.unsqueeze(2)).squeeze(), self.semantic_conv(query_feature.unsqueeze(2)).squeeze()), 0)
        support = rearrange(support, "b c h w -> b h w c")
        query = rearrange(query, "b c h w -> b h w c")

        support_feature = self.norm(support_feature)
        query_feature = self.norm(query_feature)

        support_key = self.norm(self.key_fc(support))
        support_value = self.norm(self.value_fc(support))
        query_key = self.norm(self.key_fc(query))
        query_value = self.norm(self.value_fc(query))

        support_key = rearrange(support_key, "b h w c -> b (h w) c")
        query_key = rearrange(query_key, "b h w c -> b (h w) c")

        support_value = rearrange(support_value, "b h w c -> b (h w) c")
        query_value = rearrange(query_value, "b h w c -> b (h w) c")

        support_key = rearrange(support_key, "b n_b c -> b c n_b")
        query_key = rearrange(query_key, "b n_b c -> b c n_b")

        support_dp = (support_feature @ support_key) * self.scale
        support_attention = support_dp.softmax(dim=-1)
        query_dp = (query_feature @ query_key) * self.scale
        query_attention = query_dp.softmax(dim=-1)
        
        support_value = support_attention @ support_value
        query_value = query_attention @ query_value

        support_value = support_value.reshape((support_value.shape[0], shape[2], shape[3], support_value.shape[2]))
        query_value = query_value.reshape((query_value.shape[0], shape[2], shape[3], query_value.shape[2]))
        support = rearrange(support_value, 'b h w c -> b c h w')
        query = rearrange(query_value, 'b h w c -> b c h w ')
        support_vis = support
        query_vis = query

        shape = support.shape
        support = support.reshape(self.num_shot, self.num_way, support.shape[1] , support.shape[2] , support.shape[3])
        support = torch.transpose(support, 0, 1)

        #for one shot
        if support.shape[1] == 1:
            support = support.squeeze(1)
        else:
            support = self.proxy(support)

        support = support.unsqueeze(0).repeat(self.num_query * self.num_way,1,1,1,1)
        query = query.unsqueeze(0).repeat(self.num_way, 1, 1, 1, 1)
        query = torch.transpose(query, 0, 1)

        #n_samples, channel, height, width 
        support = support.reshape(-1, support.shape[2], support.shape[3], support.shape[4])
        query = query.reshape(-1, query.shape[2], query.shape[3], query.shape[4])

        feature = None
        if isinstance(self.classifier, ConvClassifier):
            support = support.unsqueeze(1)
            query = query.unsqueeze(1)
            feature = torch.cat((support, query), 1)
        elif isinstance(self.classifier, FCClassifier):
            feature = torch.cat((support, query), 1)
        elif isinstance(self.classifier, EuclideanClassifier) or isinstance(self.classifier, CosineClassifier):
            out = self.classifier(support, query)
            out = out.view(-1, self.num_way)
            return out, visual_feature, support_vis, query_vis
        else:
            raise ("Classifier value error!")

        out = self.classifier(feature)
        out = out.view(-1, self.num_way)

        return out, visual_feature, support_vis, query_vis
