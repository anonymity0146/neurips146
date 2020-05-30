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
from scipy._lib._util import _asarray_validated

gpu = '0'
gpu = gpu.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)
os.environ['QT_QPA_PLATFORM']='offscreen'

# load model
PRETRAIN = 'none'  # 'real', 'imagenet', 'none'
TARGET_SOURCE = 'content'  # 'random', 'content'
OPTIMIZOR = 'LBFGS'  # 'Adam', 'LBFGS'
ARCHITECTURE = 'resnet50'
LOADED_ARCHITECTURE = 'resnet50_plain16'
MULTIPLE = ''

if PRETRAIN == 'imagenet':
    resnet = models.resnet50(pretrained=True)
else:
    resnet = models.__dict__[ARCHITECTURE](pretrained=False)
    if ARCHITECTURE == 'resnet50_plain16' or ARCHITECTURE == 'resnet50_plain12':
        if './checkpoints/checkpoint_pretrain_' + LOADED_ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + MULTIPLE + '.pth.tar':
            if os.path.isfile('./checkpoints/checkpoint_pretrain_' + LOADED_ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + MULTIPLE + '.pth.tar'):
                print("=> loading checkpoint '{}'".format(
                    './checkpoints/checkpoint_pretrain_' + LOADED_ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + MULTIPLE + '.pth.tar'))
                checkpoint = torch.load('./checkpoints/checkpoint_pretrain_' + LOADED_ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + MULTIPLE + '.pth.tar')
                resnet.load_state_dict(checkpoint['state_dict_m'])
            else:
                print("=> no checkpoint found at '{}'".format(
                    './checkpoints/checkpoint_pretrain_' + LOADED_ARCHITECTURE + '_' + PRETRAIN + '_' + TARGET_SOURCE + MULTIPLE + '.pth.tar'))



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


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):

    a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def get_features(image, model, layers=None):
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

    elif ARCHITECTURE == 'resnet50_small_conv' or ARCHITECTURE == 'resnet50_plain2' or ARCHITECTURE == 'resnet50_plain3' or ARCHITECTURE == 'resnet50_plain6' or ARCHITECTURE == 'resnet50_plain9' or ARCHITECTURE == 'resnet50_plain10' or ARCHITECTURE == 'resnet50_plain12' or ARCHITECTURE == 'resnet50_plain16':

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
        # x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        # x = model.bn2(x)
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
        # x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        # x = model.bn2(x)
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


    elif ARCHITECTURE == 'resnet50_plain11':

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
        features_resi = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        features_resi['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        features_resi['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers1:
                features[layers1[str(name)]] = x
                features_resi[layers1[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers2:
                features[layers2[str(name)]] = x
                features_resi[layers2[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers3:
                features[layers3[str(name)]] = x
                features_resi[layers3[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers4:
                features[layers4[str(name)]] = x
                features_resi[layers4[str(name)]] = x_resi


    elif ARCHITECTURE == 'resnet50_plain13':

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
        features_resi = {}
        x = image
        x = model.conv1(x)
        features['conv0_0'] = x
        features_resi['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
        # x = model.maxpool(x)
        x = model.conv2(x)
        features['conv0_1'] = x
        features_resi['conv0_1'] = x
        x = model.bn2(x)
        x = model.relu(x)
        x = model.maxpool(x)

        for name, layer in enumerate(model.layer1):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers1:
                features[layers1[str(name)]] = x
                features_resi[layers1[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer2):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers2:
                features[layers2[str(name)]] = x
                features_resi[layers2[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer3):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers3:
                features[layers3[str(name)]] = x
                features_resi[layers3[str(name)]] = x_resi
        x = model.maxpool(x)
        for name, layer in enumerate(model.layer4):
            x, x_resi = layer(x)
            x_resi = x_resi.detach()
            if str(name) in layers4:
                features[layers4[str(name)]] = x
                features_resi[layers4[str(name)]] = x_resi

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
        features['conv0_0'] = x
        x = model.bn1(x)
        x = model.relu(x)
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

    # elif ARCHITECTURE == 'resnet50_plain11' or ARCHITECTURE == 'resnet50_plain13':
    #     return features, features_resi
    # else:
    #     return features





def gram_matrix(input):
    b, c, d = np.shape(input)  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = np.reshape(input, (b, c * d))

    G = np.matmul(features, np.transpose(features))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return np.true_divide(G, b * c * d)


def activation_visualization(model, style, content, target, style_img_name, content_img_name, saved_log_print):

    # save main output
    if ARCHITECTURE == 'resnet50_plain11' or ARCHITECTURE == 'resnet50_plain13':
        content_features, _ = get_features(content, model)
        style_features, style_features_resi = get_features(style, model)
        target_features, _ = get_features(target, model)
        for layer, feature in style_features.items():
            feature = feature[0].cpu().data.numpy().squeeze()
            feature_stat = feature.flatten()
            # plot the statistics of feature maps
            plt.hist(feature_stat)
            plt.ylabel('number')
            plt.xlabel('feature magitude')
            if not os.path.exists('./results_new/' + ARCHITECTURE):
                os.makedirs('./results_new/' + ARCHITECTURE)
            hist_name = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_style_activation_statistics_' + layer + '_' + MULTIPLE + '.png'
            plt.savefig(hist_name)
            plt.close()



            style_gram = gram_matrix(feature)
            plt.imshow(style_gram)
            plt.colorbar()

            if not os.path.exists('./results_new/' + ARCHITECTURE):
                os.makedirs('./results_new/' + ARCHITECTURE)
            save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_GramMatrix_' + layer + '_' + MULTIPLE + '.png'
            plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
            plt.close()
            print(layer)
            print('Gram_std', np.std(style_gram))

            feature = np.mean(feature, axis=0)
            plt.imshow(feature)
            plt.colorbar()

            if not os.path.exists('./results_new/' + ARCHITECTURE):
                os.makedirs('./results_new/' + ARCHITECTURE)
            save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_featuremap_' + layer + '_' + MULTIPLE + '.png'
            plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
            plt.close()
            # print('featuremap_std', np.std(feature))

    else:

        with open(saved_log_print, 'a') as f:

            print(ARCHITECTURE, file=f)
            print(style_img_name, file=f)

            content_features = get_features(content, model)
            style_features = get_features(style, model)
            target_features = get_features(target, model)
            for layer, feature in style_features.items():
                feature = feature[0].cpu().data.numpy().squeeze()
                feature_stat = feature.flatten()
                # plot the statistics of feature maps
                plt.hist(feature_stat)
                plt.ylabel('number')
                plt.xlabel('feature magitude')
                if not os.path.exists('./results_new/' + ARCHITECTURE):
                    os.makedirs('./results_new/' + ARCHITECTURE)
                hist_name = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_style_activation_statistics_' + layer + '_' + MULTIPLE + '.png'
                plt.savefig(hist_name)
                plt.close()

                print(layer, file=f)
                feature_max = np.max(feature_stat)
                print('feature_max', feature_max, file=f)
                print('feature_std', np.std(feature_stat), file=f)
                feature_stat = softmax(feature_stat)
                cur_entropy = entropy(feature_stat)
                cur_entropy = cur_entropy / np.log(len(feature_stat.tolist()))
                print('feature_entropy', cur_entropy, file=f)

                style_gram = gram_matrix(feature)
                plt.imshow(style_gram)
                plt.colorbar()

                if not os.path.exists('./results_new/' + ARCHITECTURE):
                    os.makedirs('./results_new/' + ARCHITECTURE)
                save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_GramMatrix_' + layer + '_' + MULTIPLE + '.png'
                plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
                plt.close()

                print('Gram_max', np.max(style_gram), file=f)
                print('Gram_std', np.std(style_gram), file=f)
                # entropy
                style_gram = style_gram.flatten()
                style_gram = softmax(style_gram)
                cur_entropy = entropy(style_gram)
                cur_entropy = cur_entropy / np.log(len(style_gram.tolist()))
                print('Gram_entropy', cur_entropy, file=f)


                feature = np.mean(feature, axis=0)
                plt.imshow(feature)
                plt.colorbar()

                if not os.path.exists('./results_new/' + ARCHITECTURE):
                    os.makedirs('./results_new/' + ARCHITECTURE)
                save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_featuremap_' + layer + '_' + MULTIPLE + '.png'

                plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
                plt.close()
                # print('featuremap_std', np.std(feature))

    # save residual output
    if ARCHITECTURE == 'resnet50_plain11' or ARCHITECTURE == 'resnet50_plain13':
        content_features, _ = get_features(content, model)
        style_features, style_features_resi = get_features(style, model)
        target_features, _ = get_features(target, model)

        for layer, feature_resi in style_features_resi.items():
            feature_resi = feature_resi[0].cpu().data.numpy().squeeze()
            feature_resi_stat = feature_resi.flatten()
            # plot the statistics of feature maps
            plt.hist(feature_resi_stat)
            plt.ylabel('number')
            plt.xlabel('feature magitude')
            if not os.path.exists('./results_new/' + ARCHITECTURE):
                os.makedirs('./results_new/' + ARCHITECTURE)
            hist_name = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_style_activation_statistics_residual_' + layer + '_' + MULTIPLE + '.png'
            plt.savefig(hist_name)
            plt.close()


            feature_resi = np.mean(feature_resi, axis=0)
            plt.imshow(feature_resi)
            plt.colorbar()
            if not os.path.exists('./results_new/' + ARCHITECTURE):
                os.makedirs('./results_new/' + ARCHITECTURE)
            save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_featuremap_residual_' + layer + '_' + MULTIPLE + '.png'
            plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
            plt.close()

    # # save target output
    # for layer, feature in target_features.items():
    #     feature = feature[0].cpu().data.numpy().squeeze()
    #     feature = np.mean(feature, axis=0)
    #     feature = (feature - np.min(feature)) / np.max(feature)
    #     _, _, w, h = target.shape
    #     mask = cv2.resize(feature, (h, w))
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     cam = heatmap
    #
    #     if not os.path.exists('./results_new/' + ARCHITECTURE):
    #         os.makedirs('./results_new/' + ARCHITECTURE)
    #     save_path = './results_new/' + ARCHITECTURE + '/' + content_img_name + '_' + style_img_name + '_' + PRETRAIN + '_' + TARGET_SOURCE + '_target_activation_' + layer + '.png'
    #
    #     cv2.imwrite(save_path, np.uint8(255 * cam))



def style_transfer(model, style, content, target, style_layer_weights, content_layer_weights, style_weight, content_weight, optimizer):

    content_features = get_features(content, model)
    style_features = get_features(style, model)

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

        for i in range(1, 100):
            def closure():
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
                total_loss.backward()
                return content_weight * content_loss + style_weight * style_loss

            optimizer.step(closure)
        final_img = im_convert(target)
        return final_img





for param in resnet.parameters():
  param.requires_grad_(False)

torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device).eval()

# load style and content images combinations

style_image_list  = []
for file in glob.glob('./style_images2/*'):
    style_image_list.append(file)

content_image_list  = []
for file in glob.glob('./content_images2/*'):
    content_image_list.append(file)

saved_log_print = 'NST_resnet_variant_random_visualization_new2.txt'

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

            elif ARCHITECTURE == 'resnet50_plain4' or ARCHITECTURE == 'resnet50_plain5' or ARCHITECTURE == 'resnet50_wo_skip2' or ARCHITECTURE == 'resnet50_plain7' or ARCHITECTURE == 'resnet50_plain8' or ARCHITECTURE == 'resnet50_plain9' or ARCHITECTURE == 'resnet50_plain10':
                style_weights = {'conv0_0': 1.0,
                                 'conv1_2': 1.0,
                                 'conv2_3': 1.0,
                                 'conv3_5': 1.0,
                                 'conv4_2': 1.0}
                content_weights = 'conv1_2'
            else:
                style_weights = {'conv1_2': 1.0,
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

        style_img_name = style_image_list[i_style].split('/')
        style_img_name = style_img_name[-1].split('.')
        style_img_name = style_img_name[0]
        content_img_name = content_image_list[i_content].split('/')
        content_img_name = content_img_name[-1].split('.')
        content_img_name = content_img_name[0]

        activation_visualization(resnet, style, content, target, style_img_name, content_img_name, saved_log_print)







