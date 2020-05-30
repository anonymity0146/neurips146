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
import torch.nn.functional as F
import cv2
from scipy.stats import entropy

gpu = '2'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM']='offscreen'

# load model
PRETRAIN = 'imagenet'  # 'real', 'imagenet', 'none'
TARGET_SOURCE = 'content'  # 'random'
OPTIMIZOR = 'LBFGS'  # 'Adam', 'LBFGS'
ARCHITECTURE = 'inception_v3'


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

    elif ARCHITECTURE == 'inception_v3_softmax':
        T = 10
        softmax = nn.Softmax2d()

        features = {}
        x = image
        x = model.Conv2d_1a_3x3(x)
        features['Conv2d_1a_3x3'] = x
        x = model.Conv2d_2a_3x3(x)
        features['Conv2d_2a_3x3'] = x
        x = model.Conv2d_2b_3x3(x)
        features['Conv2d_2b_3x3'] = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_1'] = x
        x = model.Conv2d_3b_1x1(x)
        features['Conv2d_3b_1x1'] = x
        x = model.Conv2d_4a_3x3(x)
        features['Conv2d_4a_3x3'] = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_2'] = x
        x = model.Mixed_5b(x)
        # x = softmax(x / T)
        features['Mixed_5b'] = x
        x = model.Mixed_5c(x)
        # x = softmax(x / T)
        features['Mixed_5c'] = x
        x = model.Mixed_5d(x)
        # x = softmax(x / T)
        features['Mixed_5d'] = x
        x = model.Mixed_6a(x)
        features['Mixed_6a'] = x
        x = model.Mixed_6b(x)
        features['Mixed_6b'] = x
        x = model.Mixed_6c(x)
        features['Mixed_6c'] = x
        x = model.Mixed_6d(x)
        features['Mixed_6d'] = x
        x = model.Mixed_6e(x)
        features['Mixed_6e'] = x
        x = model.Mixed_7a(x)
        features['Mixed_7a'] = softmax(x / T)
        x = model.Mixed_7b(x)
        features['Mixed_7b'] = softmax(x / T)
        x = model.Mixed_7c(x)
        features['Mixed_7c'] = softmax(x / T)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features['adaptive_avg_pool2d'] = x

    elif ARCHITECTURE == 'inception_v3_softmax_v2':

        softmax = nn.Softmax2d()

        features = {}
        x = image
        x = model.Conv2d_1a_3x3(x)
        features['Conv2d_1a_3x3'] = softmax(x / temperature['Conv2d_1a_3x3'])
        x = model.Conv2d_2a_3x3(x)
        features['Conv2d_2a_3x3'] = softmax(x / temperature['Conv2d_2a_3x3'])
        x = model.Conv2d_2b_3x3(x)
        features['Conv2d_2b_3x3'] = softmax(x / temperature['Conv2d_2b_3x3'])
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_1'] = softmax(x / temperature['max_pool2d_1'])
        x = model.Conv2d_3b_1x1(x)
        features['Conv2d_3b_1x1'] = softmax(x / temperature['Conv2d_3b_1x1'])
        x = model.Conv2d_4a_3x3(x)
        features['Conv2d_4a_3x3'] = softmax(x / temperature['Conv2d_4a_3x3'])
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_2'] = softmax(x / temperature['max_pool2d_2'])
        x = model.Mixed_5b(x)
        features['Mixed_5b'] = softmax(x / temperature['Mixed_5b'])
        x = model.Mixed_5c(x)
        features['Mixed_5c'] = softmax(x / temperature['Mixed_5c'])
        x = model.Mixed_5d(x)
        features['Mixed_5d'] = softmax(x / temperature['Mixed_5d'])
        x = model.Mixed_6a(x)
        features['Mixed_6a'] = softmax(x / temperature['Mixed_6a'])
        x = model.Mixed_6b(x)
        features['Mixed_6b'] = softmax(x / temperature['Mixed_6b'])
        x = model.Mixed_6c(x)
        features['Mixed_6c'] = softmax(x / temperature['Mixed_6c'])
        x = model.Mixed_6d(x)
        features['Mixed_6d'] = softmax(x / temperature['Mixed_6d'])
        x = model.Mixed_6e(x)
        features['Mixed_6e'] = softmax(x / temperature['Mixed_6e'])
        x = model.Mixed_7a(x)
        features['Mixed_7a'] = softmax(x / temperature['Mixed_7a'])
        x = model.Mixed_7b(x)
        features['Mixed_7b'] = softmax(x / temperature['Mixed_7b'])
        x = model.Mixed_7c(x)
        features['Mixed_7c'] = softmax(x / temperature['Mixed_7c'])
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features['adaptive_avg_pool2d'] = softmax(x / temperature['adaptive_avg_pool2d'])

    else:

        features = {}
        x = image
        x = model.Conv2d_1a_3x3(x)
        features['Conv2d_1a_3x3'] = x
        x = model.Conv2d_2a_3x3(x)
        features['Conv2d_2a_3x3'] = x
        x = model.Conv2d_2b_3x3(x)
        features['Conv2d_2b_3x3'] = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_1'] = x
        x = model.Conv2d_3b_1x1(x)
        features['Conv2d_3b_1x1'] = x
        x = model.Conv2d_4a_3x3(x)
        features['Conv2d_4a_3x3'] = x
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        features['max_pool2d_2'] = x
        x = model.Mixed_5b(x)
        features['Mixed_5b'] = x
        x = model.Mixed_5c(x)
        features['Mixed_5c'] = x
        x = model.Mixed_5d(x)
        features['Mixed_5d'] = x
        x = model.Mixed_6a(x)
        features['Mixed_6a'] = x
        x = model.Mixed_6b(x)
        features['Mixed_6b'] = x
        x = model.Mixed_6c(x)
        features['Mixed_6c'] = x
        x = model.Mixed_6d(x)
        features['Mixed_6d'] = x
        x = model.Mixed_6e(x)
        features['Mixed_6e'] = x
        x = model.Mixed_7a(x)
        features['Mixed_7a'] = x
        x = model.Mixed_7b(x)
        features['Mixed_7b'] = x
        x = model.Mixed_7c(x)
        features['Mixed_7c'] = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features['adaptive_avg_pool2d'] = x

    return features


def compute_temperature(image, model):
    softmax = nn.Softmax2d()

    temperature = {}
    x = image
    x = model.Conv2d_1a_3x3(x)
    temperature['Conv2d_1a_3x3'] = output_temperature(x)
    x = model.Conv2d_2a_3x3(x)
    temperature['Conv2d_2a_3x3'] = output_temperature(x)
    x = model.Conv2d_2b_3x3(x)
    temperature['Conv2d_2b_3x3'] = output_temperature(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    temperature['max_pool2d_1'] = output_temperature(x)
    x = model.Conv2d_3b_1x1(x)
    temperature['Conv2d_3b_1x1'] = output_temperature(x)
    x = model.Conv2d_4a_3x3(x)
    temperature['Conv2d_4a_3x3'] = output_temperature(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    temperature['max_pool2d_2'] = output_temperature(x)
    x = model.Mixed_5b(x)
    temperature['Mixed_5b'] = output_temperature(x)
    x = model.Mixed_5c(x)
    temperature['Mixed_5c'] = output_temperature(x)
    x = model.Mixed_5d(x)
    temperature['Mixed_5d'] = output_temperature(x)
    x = model.Mixed_6a(x)
    temperature['Mixed_6a'] = output_temperature(x)
    x = model.Mixed_6b(x)
    temperature['Mixed_6b'] = output_temperature(x)
    x = model.Mixed_6c(x)
    temperature['Mixed_6c'] = output_temperature(x)
    x = model.Mixed_6d(x)
    temperature['Mixed_6d'] = output_temperature(x)
    x = model.Mixed_6e(x)
    temperature['Mixed_6e'] = output_temperature(x)
    x = model.Mixed_7a(x)
    temperature['Mixed_7a'] = output_temperature(x)
    x = model.Mixed_7b(x)
    temperature['Mixed_7b'] = output_temperature(x)
    x = model.Mixed_7c(x)
    temperature['Mixed_7c'] = output_temperature(x)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    temperature['adaptive_avg_pool2d'] = output_temperature(x)
    return temperature


def output_temperature(x):
    softmax = nn.Softmax2d()
    x_dis = softmax(x)
    x_dis = x_dis.detach()
    x_dis = x_dis.cpu().numpy()
    x_dis = x_dis.flatten()
    cur_entropy = entropy(x_dis)
    cur_entropy = cur_entropy / np.log(len(x_dis.tolist()))
    alpha = 1.0
    T = np.exp(-alpha * (cur_entropy - 1.0))
    return T


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
                target_features = get_features(target, model, temperatures)

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
        return final_img





if PRETRAIN == 'imagenet':
    inception = models.inception_v3(pretrained=True)
else:
    inception = models.__dict__['inception_v3'](pretrained=False)

for param in inception.parameters():
    param.requires_grad_(False)



torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception.to(device).eval()
# load style and content images combinations

style_image_list  = []
for file in glob.glob('./style_images/*'):
    style_image_list.append(file)

content_image_list  = []
for file in glob.glob('./content_images/*'):
    content_image_list.append(file)

for i_style in range(len(style_image_list)):
    for i_content in range(len(content_image_list)):

        print('processing content', i_content, ' style ', i_style)

        style = load_image(style_image_list[i_style]).to(device)
        content = load_image(content_image_list[i_content]).to(device)

        if TARGET_SOURCE == 'random':
            target = torch.randn_like(content).requires_grad_(True).to(device)
        elif TARGET_SOURCE == 'content':
            target = content.clone().requires_grad_(True).to(device)

        style_weights = {'Conv2d_1a_3x3': 1.0,
                         'Conv2d_3b_1x1': 1.0,
                         'Mixed_5b': 1.0,
                         'Mixed_6a': 1.0,
                         'Mixed_7a': 1.0
                         }

        if PRETRAIN == 'imagenet':
            content_weights = 'Mixed_5b'
        elif PRETRAIN == 'none':
            content_weights = 'Mixed_5b'

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

        final_styled = style_transfer(inception, style, content, target, style_weights, content_weights, style_weight, content_weight, optimizer)

        style_img_name = style_image_list[i_style].split('/')
        style_img_name = style_img_name[-1].split('.')
        style_img_name = style_img_name[0]
        content_img_name = content_image_list[i_content].split('/')
        content_img_name = content_img_name[-1].split('.')
        content_img_name = content_img_name[0]

        if not os.path.exists('./results_user_study/' + ARCHITECTURE):
            os.makedirs('./results_user_study/' + ARCHITECTURE)

        save_path_cv2 = './results_user_study/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_cv2.png'
        final_styled_cv2 = np.uint8(255 * final_styled)
        final_styled_cv2_bgr = final_styled_cv2[:, :, [2, 1, 0]]
        cv2.imwrite(save_path_cv2, final_styled_cv2_bgr)




if PRETRAIN == 'none':
    save_checkpoint({
        'state_dict_m': inception.cpu().state_dict(),
    }, filename='./checkpoints/checkpoint_pretrain_' + ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + '.pth.tar')

