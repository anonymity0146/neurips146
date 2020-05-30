from collections import namedtuple
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
# For getting VGG model
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
# Image transformation pipeline
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm_notebook
from torchvision import transforms, models
from transformer_net import TransformerNet
from utils import gram_matrix, recover_image, tensor_normalizer
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

gpu = '3'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM']='offscreen'

ARCHITECTURE = 'resnet50_softmax'
STYLE_FIELD = 'picasso'
PRETRAIN = 'random'

# SEED = 1080
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     kwargs = {'num_workers': 4, 'pin_memory': True}
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')
#     kwargs = {}
kwargs = {'num_workers': 4, 'pin_memory': True}

IMAGE_SIZE = 256
BATCH_SIZE = 4
DATASET = "./coco/"
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                tensor_normalizer()])
# http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
train_dataset = datasets.ImageFolder(DATASET, transform)
# http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

LossOutput = namedtuple("LossOutput", ["conv0_0", "conv1_2", "conv2_3", "conv3_5", "conv4_2"])


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, resnet_model):
        super(LossNetwork, self).__init__()
        self.model = resnet_model
        self.layers1 = {'2': 'conv1_2'}
        self.layers2 = {'3': 'conv2_3'}
        self.layers3 = {'5': 'conv3_5'}
        self.layers4 = {'2': 'conv4_2'}


    def forward(self, x):

        if ARCHITECTURE == 'resnet50_softmax':
            output = {}

            softmax = nn.Softmax2d()
            T = 1.0
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            output['conv0_0'] = x
            # x = softmax(x / T)
            x = self.model.maxpool(x)

            for name, layer in enumerate(self.model.layer1):
                x = layer(x)
                x = softmax(x / T)
                if str(name) in self.layers1:
                    output[self.layers1[str(name)]] = x
            for name, layer in enumerate(self.model.layer2):
                x = layer(x)
                x = softmax(x / T)
                if str(name) in self.layers2:
                    output[self.layers2[str(name)]] = x
            for name, layer in enumerate(self.model.layer3):
                x = layer(x)
                x = softmax(x / T)
                if str(name) in self.layers3:
                    output[self.layers3[str(name)]] = x
            for name, layer in enumerate(self.model.layer4):
                x = layer(x)
                x = softmax(x / T)
                if str(name) in self.layers4:
                    output[self.layers4[str(name)]] = x
        else:

            output = {}
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            output['conv0_0'] = x
            x = self.model.maxpool(x)

            for name, layer in enumerate(self.model.layer1):
                x = layer(x)
                if str(name) in self.layers1:
                    output[self.layers1[str(name)]] = x
            for name, layer in enumerate(self.model.layer2):
                x = layer(x)
                if str(name) in self.layers2:
                    output[self.layers2[str(name)]] = x
            for name, layer in enumerate(self.model.layer3):
                x = layer(x)
                if str(name) in self.layers3:
                    output[self.layers3[str(name)]] = x
            for name, layer in enumerate(self.model.layer4):
                x = layer(x)
                if str(name) in self.layers4:
                    output[self.layers4[str(name)]] = x

        return LossOutput(**output)


# class LossNetwork(torch.nn.Module):
#     def __init__(self, vgg_model):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg_model.features
#         self.layer_name_mapping = {
#             '3': "relu1_2",
#             '8': "relu2_2",
#             '15': "relu3_3",
#             '22': "relu4_3"
#         }
#
#     def forward(self, x):
#         output = {}
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layer_name_mapping:
#                 output[self.layer_name_mapping[name]] = x
#         return LossOutput(**output)



resnet_model = models.resnet50(pretrained=False)
if torch.cuda.is_available():
    resnet_model.cuda()
loss_network = LossNetwork(resnet_model)
loss_network.eval()



def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)

save_checkpoint({
    'state_dict_m': resnet_model.cpu().state_dict(),
}, filename='./transformer_ckpts/checkpoint_pretrain_' + ARCHITECTURE + '_' + PRETRAIN + '_' + STYLE_FIELD + '2.pth.tar')
resnet_model.cuda()
loss_network = LossNetwork(resnet_model)
loss_network.eval()
del resnet_model

STYLE_IMAGE = "style_images/picasso.jpg"
style_img = Image.open(STYLE_IMAGE).convert('RGB')
style_img_tensor = transforms.Compose([
    transforms.ToTensor(),
    tensor_normalizer()]
)(style_img).unsqueeze(0)
# assert np.sum(style_img - recover_image(style_img_tensor.numpy())[0].astype(np.uint8)) < 3 * style_img_tensor.size()[2] * style_img_tensor.size()[3]
if torch.cuda.is_available():
    style_img_tensor = style_img_tensor.cuda()

plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])

plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])

# http://pytorch.org/docs/master/notes/autograd.html#volatile
style_loss_features = loss_network(Variable(style_img_tensor, volatile=True))
gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features]

# style_loss_features._fields
#
# np.mean(gram_style[3].data.cpu().numpy())
#
# np.mean(style_loss_features[3].data.cpu().numpy())
#
# gram_style[0].numel()

def save_debug_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)

transformer = TransformerNet()
mse_loss = torch.nn.MSELoss()
# l1_loss = torch.nn.L1Loss()
if torch.cuda.is_available():
    transformer.cuda()

CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 1e11
LOG_INTERVAL = 200
REGULARIZATION = 1e-7

LR = 1e-4
optimizer = Adam(transformer.parameters(), LR)
transformer.train()
for epoch in range(3):
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_reg_loss = 0.
    count = 0
    for batch_id, (x, _) in tqdm_notebook(enumerate(train_loader), total=len(train_loader)):
        n_batch = len(x)
        count += n_batch
        optimizer.zero_grad()
        x = Variable(x)
        if torch.cuda.is_available():
            x = x.cuda()

        y = transformer(x)
        xc = Variable(x.data, volatile=True)

        features_y = loss_network(y)
        features_xc = loss_network(xc)

        f_xc_c = Variable(features_xc[0].data, requires_grad=False)

        content_loss = CONTENT_WEIGHT * mse_loss(features_y[0], f_xc_c)

        reg_loss = REGULARIZATION * (
            torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        style_loss = 0.
        for m in range(len(features_y)):
            gram_s = gram_style[m]
            gram_y = gram_matrix(features_y[m])
            style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s.expand_as(gram_y))

        total_loss = content_loss + style_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()
        agg_reg_loss += reg_loss.item()

        if (batch_id + 1) % LOG_INTERVAL == 0:
            mesg = "{} [{}/{}] content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                        time.ctime(), count, len(train_dataset),
                        agg_content_loss / LOG_INTERVAL,
                        agg_style_loss / LOG_INTERVAL,
                        agg_reg_loss / LOG_INTERVAL,
                        (agg_content_loss + agg_style_loss + agg_reg_loss) / LOG_INTERVAL
                    )
            print(mesg)
            agg_content_loss = 0
            agg_style_loss = 0
            agg_reg_loss = 0
            transformer.eval()
            y = transformer(x)
            save_debug_image(x.data, y.data, "debug/{}_{}.png".format(epoch, count))
            transformer.train()

import glob
fnames = glob.glob(DATASET + "/*/*")
len(fnames)

transformer = transformer.eval()

img = Image.open(fnames[40]).convert('RGB')
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                tensor_normalizer()])
img_tensor = transform(img).unsqueeze(0)
if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()

img_output = transformer(Variable(img_tensor, volatile=True))
plt.imshow(recover_image(img_tensor.cpu().numpy())[0])

Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])

save_model_path = "model_udnie_random_resnet_softmax.pth"
torch.save(transformer.state_dict(), save_model_path)

transformer.load_state_dict(torch.load(save_model_path))

img = Image.open("content_images/amber.jpg").convert('RGB')
transform = transforms.Compose([transforms.ToTensor(),
                                tensor_normalizer()])
img_tensor = transform(img).unsqueeze(0)
print(img_tensor.size())
if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()

img_output = transformer(Variable(img_tensor, volatile=True))
plt.imshow(recover_image(img_tensor.cpu().numpy())[0])

plt.imshow(recover_image(img_output.data.cpu().numpy())[0])

output_img = Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])
output_img.save("amber_resnet_softmax_random.png")

