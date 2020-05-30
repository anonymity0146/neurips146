from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
import torch.nn as nn
import os
import glob
import models as models
import cv2

gpu = '3'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM']='offscreen'

# load model
PRETRAIN = 'none'  # 'imagenet', 'none'
TARGET_SOURCE = 'content'  # 'random', 'content'
OPTIMIZOR = 'LBFGS'  # 'Adam', 'LBFGS'
ARCHITECTURE = 'vgg19'

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image



def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)

def get_features(image, model, layers=None):

    if ARCHITECTURE == 'vgg19_adding_conv1x1_f':
        layers = {'0': 'conv1_1', '7': 'conv2_1',
                  '9': 'conv2_2',
                  '14': 'conv3_1',
                  '25': 'conv4_1',
                  '27': 'conv4_2',
                  '36': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_adding_conv1x1_g':
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '9': 'conv2_2',
                  '14': 'conv3_1',
                  '31': 'conv4_1',
                  '35': 'conv4_2',
                  '48': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_adding_conv1x1_h':


        layers = {'0': 'conv1_1', '7': 'conv2_1',
                  '11': 'conv2_2',
                  '18': 'conv3_1',
                  '37': 'conv4_1',
                  '41': 'conv4_2',
                  '56': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_adding_conv1x1_h_removing_1conv3x3':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '9': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet0':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet2':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet3':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet4':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet5':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet6':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet7':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet8':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet9':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet10':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet11':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet12':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet13':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet14':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet15':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet16':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x
                
    elif ARCHITECTURE == 'vgg19_started_resnet17':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet18':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet19':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet20':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet21':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'vgg19_started_resnet22':


        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '13': 'conv2_2',
                  '16': 'conv3_1',
                  '35': 'conv4_1',
                  '39': 'conv4_2',
                  '54': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x


    elif ARCHITECTURE == 'vgg19_removing_1conv3x3':
        layers = {'0': 'conv1_1', '3': 'conv2_1',
                  '5': 'conv2_2',
                  '8': 'conv3_1',
                  '17': 'conv4_1',
                  '19': 'conv4_2',
                  '26': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x


    else:
        layers = {'0': 'conv1_1', '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x


    return features


# def gram_matrix(tensor):
#     _, n_filters, h, w = tensor.size()
#     tensor = tensor.view(n_filters, h * w)
#     gram = torch.mm(tensor, tensor.t())
#
#     return gram

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def style_transfer(model, style, content, target, style_layer_weights, content_layer_weights, style_weight, content_weight, optimizer):

    content_features = get_features(content, model)
    style_features = get_features(style, model)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    if OPTIMIZOR == 'Adam':
        for i in range(1, 400):
            optimizer.zero_grad()
            target_features = get_features(target, model)

            content_loss = torch.mean((target_features[content_layer_weights] -
                                       content_features[content_layer_weights]) ** 2)
            style_loss = 0
            for layer in style_layer_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = style_layer_weights[layer] * torch.mean(
                    (target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()
        final_img = im_convert(target)
        return final_img

    elif OPTIMIZOR == 'LBFGS':

        run = [0]
        while run[0] <= 3000:
            def closure():
                optimizer.zero_grad()
                target_features = get_features(target, model)

                content_loss = torch.mean((target_features[content_layer_weights] -
                                           content_features[content_layer_weights]) ** 2)
                style_loss = 0
                for layer in style_layer_weights:
                    target_feature = target_features[layer]
                    target_gram = gram_matrix(target_feature)
                    # _, d, h, w = target_feature.shape
                    style_gram = style_grams[layer]
                    layer_style_loss = style_layer_weights[layer] * torch.mean(
                        (target_gram - style_gram) ** 2)
                    style_loss += style_weight * layer_style_loss

                total_loss = content_weight * content_loss + style_loss
                total_loss.backward()
                run[0] += 1
                return content_weight * content_loss + style_loss

            optimizer.step(closure)
        final_img = im_convert(target)

        # final style loss
        target_features = get_features(target, model)
        style_loss = 0
        for layer in style_layer_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_layer_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2)
            style_loss += layer_style_loss


        return final_img, style_loss.item()





if PRETRAIN == 'imagenet':
    vgg = models.vgg19(pretrained=True)
else:
    vgg = models.__dict__[ARCHITECTURE](pretrained=False)





for param in vgg.parameters():
  param.requires_grad_(False)

# for i, layer in enumerate(vgg.features):
#   if isinstance(layer, torch.nn.MaxPool2d):
#     vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device).eval()

# load style and content images combinations

style_image_list  = []
for file in glob.glob('./style_images/*'):
    style_image_list.append(file)

content_image_list  = []
for file in glob.glob('./content_images/*'):
    content_image_list.append(file)


vgg_style_losses = np.load('vgg_style_losses.npy')
vgg_style_losses = vgg_style_losses[()]
each_style_loss = np.zeros((len(style_image_list),len(content_image_list)))
for i_style in range(len(style_image_list)):
    for i_content in range(len(content_image_list)):

        print('processing content', i_content, ' style ', i_style)

        style = load_image(style_image_list[i_style]).to(device)
        content = load_image(content_image_list[i_content]).to(device)

        if TARGET_SOURCE == 'random':
            target = torch.randn_like(content).requires_grad_(True).to(device)
        elif TARGET_SOURCE == 'content':
            target = content.clone().requires_grad_(True).to(device)

        if ARCHITECTURE == 'vgg19_adding_conv1x1_h':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_adding_conv1x1_h_removing_1conv3x3':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet0':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet2':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet3':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet4':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet5':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet6':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet7':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet8':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet9':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet10':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet11':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet12':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet13':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet14':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet15':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet16':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet17':
            style_weights = {'conv1_1': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet18':
            style_weights = {'conv1_2': 1.0,
                             'conv2_4': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet19':
            style_weights = {'conv1_2': 1.0,
                             'conv2_4': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet20':
            style_weights = {'conv1_2': 1.0,
                             'conv2_4': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet21':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}
        elif ARCHITECTURE == 'vgg19_started_resnet22':
            style_weights = {'conv1_2': 1.0,
                             'conv2_2': 1.0,
                             'conv3_4': 1.0,
                             'conv4_4': 1.0,
                             'conv5_4': 1.0}

        else:
            style_weights = {'conv1_1': 1.0,
                             'conv2_1': 1.0,
                             'conv3_1': 1.0,
                             'conv4_1': 1.0,
                             'conv5_1': 1.0}

        if PRETRAIN == 'imagenet':
            content_weights = 'conv4_2'
        elif PRETRAIN == 'none' or PRETRAIN == 'real':
            content_weights = 'conv4_2'

        if TARGET_SOURCE == 'content':
            content_weight = 1
            style_weight = 4e10
        elif TARGET_SOURCE == 'random':
            content_weight = 1e3
            style_weight = 1

        if OPTIMIZOR == 'Adam':
            optimizer = optim.Adam([target], lr=0.01)
        elif OPTIMIZOR == 'LBFGS':
            optimizer = optim.LBFGS([target])

        final_styled, style_loss = style_transfer(vgg, style, content, target, style_weights, content_weights, style_weight, content_weight, optimizer)

        each_style_loss[i_style, i_content] = style_loss


        style_img_name = style_image_list[i_style].split('/')
        style_img_name = style_img_name[-1].split('.')
        style_img_name = style_img_name[0]
        content_img_name = content_image_list[i_content].split('/')
        content_img_name = content_img_name[-1].split('.')
        content_img_name = content_img_name[0]

        if not os.path.exists('./results_user_study/' + ARCHITECTURE):
            os.makedirs('./results_user_study/' + ARCHITECTURE)

        save_path = './results_user_study/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '.png'
        # save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_wo_conv0_0.png'

        save_path_cv2 = './results_user_study/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_cv2.png'
        final_styled_cv2 = np.uint8(255 * final_styled)
        final_styled_cv2_bgr = final_styled_cv2[:, :, [2, 1, 0]]
        cv2.imwrite(save_path_cv2, final_styled_cv2_bgr)





if PRETRAIN == 'none':
    save_checkpoint({
        'state_dict_m': vgg.cpu().state_dict(),
    }, filename='./checkpoints/checkpoint_pretrain_' + ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + '.pth.tar')
