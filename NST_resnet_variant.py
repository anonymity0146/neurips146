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
from scipy.stats import entropy

gpu = '2'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM']='offscreen'

# load model
PRETRAIN = 'none'  # 'real', 'imagenet', 'none'
TARGET_SOURCE = 'content'  # 'random', 'content'
OPTIMIZOR = 'LBFGS'  # 'Adam', 'LBFGS'
ARCHITECTURE = 'resnet50_softmax_v2'

if PRETRAIN == 'imagenet':
    resnet = models.resnet50(pretrained=True)
else:
    resnet = models.__dict__[ARCHITECTURE](pretrained=False)

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

def get_features(image, model, temperature, layers=None):
    # if layers is None:
    #     layers = {'0': 'conv1_1', '5': 'conv2_1',
    #               '10': 'conv3_1',
    #               '19': 'conv4_1',
    #               '21': 'conv4_2',  ## content layer
    #               '28': 'conv5_1'}
    # features = {}
    # x = image
    # for name, layer in enumerate(model.features):
    #     x = layer(x)
    #     if str(name) in layers:
    #         features[layers[str(name)]] = x

    if ARCHITECTURE == 'resnet50_started_vgg':
        if layers is None:
            layers = {'0': 'conv1',
                      '2': 'conv2',
                      '5': 'conv3',
                      '7': 'conv4',
                      '10': 'conv5',
                      '12': 'conv6',
                      '15': 'conv7',
                      '17': 'conv8'
                      }
        features = {}
        x = image
        for name, layer in enumerate(model.features):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

    elif ARCHITECTURE == 'resnet50_small_conv' or ARCHITECTURE == 'resnet50_plain2' or ARCHITECTURE == 'resnet50_plain3' or ARCHITECTURE == 'resnet50_plain6' or ARCHITECTURE == 'resnet50_plain9' or ARCHITECTURE == 'resnet50_plain10' or ARCHITECTURE == 'resnet50_plain12' or ARCHITECTURE == 'resnet50_plain16' or ARCHITECTURE == 'resnet50_plain17' or ARCHITECTURE == 'resnet50_plain18' or ARCHITECTURE == 'resnet50_plain20' or ARCHITECTURE == 'resnet50_plain21':

        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }

        features = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x


    elif ARCHITECTURE == 'resnet50_plain14' or ARCHITECTURE == 'resnet50_plain15':

        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1'
                       }

        features = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x

    elif ARCHITECTURE == 'resnet50_plain19':

        if layers is None:
            layers1 = {'0': 'conv1_0'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1'
                       }

        features = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x


    elif ARCHITECTURE == 'resnet50_plain4' or ARCHITECTURE == 'resnet50_plain5' or ARCHITECTURE == 'resnet50_plain8':
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1'
                       }

        features = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x

    # elif ARCHITECTURE == 'resnet50_softmax2':
    #     if layers is None:
    #         layers1 = {'0': 'conv1_0',
    #                    '1': 'conv1_1',
    #                    '2': 'conv1_2'
    #                    }
    #         layers2 = {'0': 'conv2_0',
    #                    '1': 'conv2_1',
    #                    '2': 'conv2_2',
    #                    '3': 'conv2_3'
    #                    }
    #         layers3 = {'0': 'conv3_0',
    #                    '1': 'conv3_1',
    #                    '2': 'conv3_2',
    #                    '3': 'conv3_3',
    #                    '4': 'conv3_4',
    #                    '5': 'conv3_5'
    #                    }
    #         layers4 = {'0': 'conv4_0',
    #                    '1': 'conv4_1',
    #                    '2': 'conv4_2'
    #                    }
    #
    #     features = {}
    #     x = image
    #     x = model.conv1(x)
    #
    #     x = model.bn1(x)
    #     x = model.relu(x)
    #     features['conv0_0'] = x
    #     x = model.maxpool(x)
    #     softmax = nn.Softmax2d()
    #     T = 100
    #
    #     for name, layer in enumerate(model.layer1):
    #         x = layer(x)
    #         if str(name) in layers1:
    #             features[layers1[str(name)]] = softmax(x / T)
    #     for name, layer in enumerate(model.layer2):
    #         x = layer(x)
    #         if str(name) in layers2:
    #             features[layers2[str(name)]] = softmax(x / T)
    #     for name, layer in enumerate(model.layer3):
    #         x = layer(x)
    #         if str(name) in layers3:
    #             features[layers3[str(name)]] = softmax(x / T)
    #     for name, layer in enumerate(model.layer4):
    #         x = layer(x)
    #         if str(name) in layers4:
    #             features[layers4[str(name)]] = softmax(x / T)
    elif ARCHITECTURE == 'resnet50_softmax_lastlayer':
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }

        features = {}
        x = image
        x = model.conv1(x)

        x = model.bn1(x)
        x = model.relu(x)
        features['conv0_0'] = x
        x = model.maxpool(x)
        features['conv0_1'] = x

        softmax = nn.Softmax2d()
        T = 100

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = softmax(x / T)
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = softmax(x / T)

    elif ARCHITECTURE == 'resnet50_softmax_v2':
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }
        softmax = nn.Softmax2d()
        features = {}
        x = image
        x = model.conv1(x)

        x = model.bn1(x)
        x = model.relu(x)
        features['conv0_0'] = softmax(x / temperature['conv0_0'])
        x = model.maxpool(x)
        features['conv0_1'] = softmax(x / temperature['conv0_1'])



        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = softmax(x / temperature[layers1[str(name)]])
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = softmax(x / temperature[layers2[str(name)]])
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = softmax(x / temperature[layers3[str(name)]])
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = softmax(x / temperature[layers4[str(name)]])

    else:
        if layers is None:
            layers1 = {'0': 'conv1_0',
                       '1': 'conv1_1',
                       '2': 'conv1_2'
                       }
            layers2 = {'0': 'conv2_0',
                       '1': 'conv2_1',
                       '2': 'conv2_2',
                       '3': 'conv2_3'
                       }
            layers3 = {'0': 'conv3_0',
                       '1': 'conv3_1',
                       '2': 'conv3_2',
                       '3': 'conv3_3',
                       '4': 'conv3_4',
                       '5': 'conv3_5'
                       }
            layers4 = {'0': 'conv4_0',
                       '1': 'conv4_1',
                       '2': 'conv4_2'
                       }

        features = {}
        x = image
        x = model.conv1(x)

        x = model.bn1(x)
        x = model.relu(x)
        features['conv0_0'] = x
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x = layer(x)
            if str(name) in layers1:
                features[layers1[str(name)]] = x
        for name, layer in enumerate(model.layer2):
            x = layer(x)
            if str(name) in layers2:
                features[layers2[str(name)]] = x
        for name, layer in enumerate(model.layer3):
            x = layer(x)
            if str(name) in layers3:
                features[layers3[str(name)]] = x
        for name, layer in enumerate(model.layer4):
            x = layer(x)
            if str(name) in layers4:
                features[layers4[str(name)]] = x

    return features

def compute_temperature(image, model):

    layers1 = {'0': 'conv1_0',
               '1': 'conv1_1',
               '2': 'conv1_2'
               }
    layers2 = {'0': 'conv2_0',
               '1': 'conv2_1',
               '2': 'conv2_2',
               '3': 'conv2_3'
               }
    layers3 = {'0': 'conv3_0',
               '1': 'conv3_1',
               '2': 'conv3_2',
               '3': 'conv3_3',
               '4': 'conv3_4',
               '5': 'conv3_5'
               }
    layers4 = {'0': 'conv4_0',
               '1': 'conv4_1',
               '2': 'conv4_2'
               }
    softmax = nn.Softmax2d()

    temperature = {}
    x = image
    x = model.conv1(x)

    x = model.bn1(x)
    x = model.relu(x)

    x_dis = softmax(x)
    x_dis = x_dis.detach()
    x_dis = x_dis.cpu().numpy()
    x_dis = x_dis.flatten()
    cur_entropy = entropy(x_dis)
    cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
    alpha = 1.0
    T = np.exp(-alpha * (cur_entropy - 1.0))

    temperature['conv0_0'] = T
    x = model.maxpool(x)

    x_dis = softmax(x)
    x_dis = x_dis.detach()
    x_dis = x_dis.cpu().numpy()
    x_dis = x_dis.flatten()
    cur_entropy = entropy(x_dis)
    cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
    alpha = 1.0
    T = np.exp(-alpha * (cur_entropy - 1.0))

    temperature['conv0_1'] = T


    for name, layer in enumerate(model.layer1):
        x = layer(x)
        if str(name) in layers1:
            x_dis = softmax(x)
            x_dis = x_dis.detach()
            x_dis = x_dis.cpu().numpy()
            x_dis = x_dis.flatten()
            cur_entropy = entropy(x_dis)
            cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
            alpha = 1.0
            T = np.exp(-alpha * (cur_entropy - 1.0))
            temperature[layers1[str(name)]] = T
    for name, layer in enumerate(model.layer2):
        x = layer(x)
        if str(name) in layers2:
            x_dis = softmax(x)
            x_dis = x_dis.detach()
            x_dis = x_dis.cpu().numpy()
            x_dis = x_dis.flatten()
            cur_entropy = entropy(x_dis)
            cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
            alpha = 1.0
            T = np.exp(-alpha * (cur_entropy - 1.0))
            temperature[layers2[str(name)]] = T
    for name, layer in enumerate(model.layer3):
        x = layer(x)
        if str(name) in layers3:
            x_dis = softmax(x)
            x_dis = x_dis.detach()
            x_dis = x_dis.cpu().numpy()
            x_dis = x_dis.flatten()
            cur_entropy = entropy(x_dis)
            cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
            alpha = 1.0
            T = np.exp(-alpha * (cur_entropy - 1.0))
            temperature[layers3[str(name)]] = T
    for name, layer in enumerate(model.layer4):
        x = layer(x)
        if str(name) in layers4:
            x_dis = softmax(x)
            x_dis = x_dis.detach()
            x_dis = x_dis.cpu().numpy()
            x_dis = x_dis.flatten()
            cur_entropy = entropy(x_dis)
            cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
            alpha = 1.0
            T = np.exp(-alpha * (cur_entropy - 1.0))
            temperature[layers4[str(name)]] = T
    return temperature

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
    temperatures = compute_temperature(style, model)

    content_features = get_features(content, model, temperatures)
    style_features = get_features(style, model, temperatures)

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    if OPTIMIZOR == 'Adam':
        for i in range(1, 500):
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
                target_features = get_features(target, model, temperatures)

                content_loss = torch.mean((target_features[content_layer_weights] -
                                           content_features[content_layer_weights]) ** 2)
                style_loss = 0
                for layer in style_layer_weights:
                    target_feature = target_features[layer]
                    target_gram = gram_matrix(target_feature)
                    # _, d, h, w = target_feature.shape
                    style_gram = style_grams[layer]

                    # if ARCHITECTURE == 'resnet50_softmax2':
                    #     softmax = nn.Softmax(dim=1)
                    #     T = 100
                    #     target_gram = target_gram.view(1, -1)
                    #     style_gram = style_gram.view(1, -1)
                    #     target_gram = softmax(target_gram / T)
                    #     style_gram = softmax(style_gram / T)

                    layer_style_loss = style_layer_weights[layer] * torch.mean(
                        (target_gram - style_gram) ** 2)
                    style_loss += style_weight * layer_style_loss

                total_loss = content_weight * content_loss + style_loss
                total_loss.backward()
                
                if run[0] % 500 == 0:
                    print("run {}:".format(run))
                    print('total Loss : {:4f}'.format(total_loss.item()))
                
                run[0] += 1
                return content_weight * content_loss + style_loss

            optimizer.step(closure)
        final_img = im_convert(target)
        # final style loss
        target_features = get_features(target, model, temperatures)
        style_loss = 0
        for layer in style_layer_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_layer_weights[layer] * torch.mean(
                (target_gram - style_gram) ** 2)
            style_loss += layer_style_loss


        return final_img, style_loss.item()









for param in resnet.parameters():
  param.requires_grad_(False)

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device).eval()



# load style and content images combinations

style_image_list  = []
for file in glob.glob('./style_images/*'):
    style_image_list.append(file)

content_image_list  = []
for file in glob.glob('./content_images/*'):
    content_image_list.append(file)

resnet_style_losses = np.load('resnet_style_losses.npy')
resnet_style_losses = resnet_style_losses[()]
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

        if ARCHITECTURE == 'resnet50_started_vgg':
            style_weights = {'conv2': 1.0,
                             'conv4': 1.0,
                             'conv6': 1.0,
                             'conv8': 1.0}
            content_weights = 'conv6'
        else:
            if ARCHITECTURE == 'resnet50_plain3' or ARCHITECTURE == 'resnet50_plain6':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_0': 1.0,
                                 'conv2_0': 1.0,
                                 'conv3_0': 1.0,
                                 'conv4_0': 1.0}
                content_weights = 'conv1_1'

            elif ARCHITECTURE == 'resnet50_plain7' or ARCHITECTURE == 'resnet50_plain8' or ARCHITECTURE == 'resnet50_plain9' or ARCHITECTURE == 'resnet50_plain10' or ARCHITECTURE == 'resnet50_plain12' or ARCHITECTURE == 'resnet50_plain16' or ARCHITECTURE == 'resnet50_plain17' or ARCHITECTURE == 'resnet50_plain18':
                style_weights = {'conv0_1': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_plain14' or ARCHITECTURE == 'resnet50_plain15' or ARCHITECTURE == 'resnet50_plain4' or ARCHITECTURE == 'resnet50_plain5':
                style_weights = {'conv0_1': 1.0,
                                 'conv1_1': 1.0,
                                 'conv2_1': 1.0,
                                 'conv3_1': 1.0,
                                 'conv4_1': 1.0}
                content_weights = 'conv3_1'
            elif ARCHITECTURE == 'resnet50_plain19':
                style_weights = {'conv0_1': 1.0,
                                 'conv1_0': 1.0,
                                 'conv2_1': 1.0,
                                 'conv3_1': 1.0,
                                 'conv4_1': 1.0}
                content_weights = 'conv3_1'
            elif ARCHITECTURE == 'resnet50_plain20':
                style_weights = {'conv0_1': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_plain21':
                style_weights = {'conv0_1': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_wo_skip2':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_wo_skip':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_decay':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'

            elif ARCHITECTURE == 'resnet50_softmax_lastlayer':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'
            elif ARCHITECTURE == 'resnet50_softmax_v2':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'


            else:
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv3_5'




        content_weight = 1
        style_weight = 1e17

        if OPTIMIZOR == 'Adam':
            optimizer = optim.Adam([target], lr=0.01)
        elif OPTIMIZOR == 'LBFGS':
            optimizer = optim.LBFGS([target])

        final_styled, style_loss = style_transfer(resnet, style, content, target, style_weights, content_weights, style_weight, content_weight, optimizer)
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


        # fig = plt.figure()
        # plt.imshow(final_styled)
        # plt.axis('off')
        # plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0, dpi=1000)
        # plt.close()

resnet_style_losses[ARCHITECTURE] = each_style_loss
np.save('resnet_style_losses.npy', resnet_style_losses)

if PRETRAIN == 'none':
    save_checkpoint({
        'state_dict_m': resnet.cpu().state_dict(),
    }, filename='./checkpoints/checkpoint_pretrain_' + ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + '.pth.tar')
