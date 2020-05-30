import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
from libs.models import encoder5 as loss_network
import models as models

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                    help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--stylePath", default="./wikiArt/",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="./coco/train2014/",
                    help='path to MSCOCO dataset')
parser.add_argument("--outf", default="trainingOutput_pretrained_resnet/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="conv3_5",
                    help='layers for content')
parser.add_argument("--style_layers", default="conv0_0,conv1_2,conv2_3,conv3_5,conv4_2",
                    help='layers for style')
parser.add_argument("--batchSize", type=int,default=8,
                    help='batch size')
parser.add_argument("--niter", type=int,default=100000,
                    help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--lr", type=float, default=1e-4,
                    help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0,
                    help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02,
                    help='style loss weight')
parser.add_argument("--log_interval", type=int, default=500,
                    help='log interval')
parser.add_argument("--gpu_id", type=int, default='3',
                    help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000,
                    help='checkpoint save interval')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
if(opt.cuda):
    torch.cuda.set_device(opt.gpu_id)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True
print_options(opt)

################# DATA #################
content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize)
content_loader_ = torch.utils.data.DataLoader(dataset     = content_dataset,
                                              batch_size  = opt.batchSize,
                                              shuffle     = True,
                                              num_workers = 1,
                                              drop_last   = True)
content_loader = iter(content_loader_)
style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                            batch_size  = opt.batchSize,
                                            shuffle     = True,
                                            num_workers = 1,
                                            drop_last   = True)
style_loader = iter(style_loader_)

################# MODEL #################
if(opt.layer == 'r31'):
    matrix = MulLayer('r31')
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    matrix = MulLayer('r41')
    vgg = encoder4()
    dec = decoder4()
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))

for param in vgg.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,
                          opt.content_layers,
                          opt.style_weight,
                          opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)

for iteration in range(1,opt.niter+1):
    optimizer.zero_grad()
    try:
        content,_ = content_loader.next()
    except IOError:
        content,_ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content,_ = content_loader.next()
    except:
        continue

    try:
        style,_ = style_loader.next()
    except IOError:
        style,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,_ = style_loader.next()
    except:
        continue

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # forward
    sF = vgg(styleV)
    cF = vgg(contentV)

    if(opt.layer == 'r41'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature,transmatrix = matrix(cF,sF)
    transfer = dec(feature)

    PRETRAIN = 'imagenet'  # 'real', 'imagenet', 'none'
    ARCHITECTURE = 'resnet50'
    if PRETRAIN == 'imagenet':
        resnet = models.resnet50(pretrained=True)
    else:
        resnet = models.__dict__[ARCHITECTURE](pretrained=False)

    for param in resnet.parameters():
        param.requires_grad_(False)
    resnet.cuda()
    resnet.eval()

    def get_features(image, model, layers=None):

        if ARCHITECTURE == 'resnet50_softmax_lastlayer':
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

    sF_loss = get_features(styleV, resnet)

    cF_loss = get_features(contentV, resnet)
    tF = get_features(transfer, resnet)
    loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)

    # backward & optimization
    loss.backward()
    optimizer.step()
    print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f Learng Rate is %.6f'%
         (opt.niter,iteration,loss,contentLoss,styleLoss,optimizer.param_groups[0]['lr']))

    adjust_learning_rate(optimizer,iteration)

    if((iteration) % opt.log_interval == 0):
        transfer = transfer.clamp(0,1)
        concat = torch.cat((content,style,transfer.cpu()),dim=0)
        vutils.save_image(concat,'%s/%d.png'%(opt.outf,iteration),normalize=True,scale_each=True,nrow=opt.batchSize)

    if(iteration > 0 and (iteration) % opt.save_interval == 0):
        torch.save(matrix.state_dict(), '%s/%s_resnet_imagenet.pth' % (opt.outf,opt.layer))
