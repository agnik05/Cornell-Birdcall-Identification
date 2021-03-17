import torch.nn as nn
import torchvision.models as models

class ResNeXt50(nn.Module):

    def __init__(self, pretrained, num_classes=264):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

def get_model(config):
    model_config = config['model']
    model_name = model_config['name']
    model_params = model_config['params']

    if model_name == 'resnext50':
        model = ResNeXt50(**model_params)

    return model
