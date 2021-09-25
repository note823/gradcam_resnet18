from config import cfg

import torch
from torch.autograd import Variable
from torchvision import models

import numpy as np
import os
import matplotlib.pyplot as plt
from urllib import request
from importlib import import_module


cfg_data_root = cfg.DATA_DIR
cfg_model = cfg.MODEL
cfg_layer_name = cfg.LAYER_NAME
cfg_layer_idx = cfg.LAYER_SEQUENTIAL_IDX
cfg_operator = cfg.OPERATOR
cfg_class_dir = cfg.IMAGET_CLASS_DIR
cfg_mean = cfg.MEAN
cfg_std = cfg.STD
cfg_resize = cfg.RESIZE
cfg_dataset = cfg.DATASET
cfg_transform = cfg.TRANSFORM


def image_tensor_to_numpy(tensor_image):
    if type(tensor_image) == np.ndarray:
        return tensor_image

    if type(tensor_image) == Variable:
        tensor_image = tensor_image.data

    np_img = tensor_image.detach().cpu().numpy()

    if len(np_img.shape) == 3:
        np_img = np_img.upsqueeze(0)

    np_img = np_img.transpose(0, 2, 3, 1)

    return np_img


def normalize(tensor):
    x = tensor - tensor.min()
    x = x / (x.max() + 1e-9)
    return x


def show_image(label, img_np, grad_CAM):
    plt.figure(figsize=(8, 8))
    plt.title(label)
    plt.imshow(img_np)
    plt.imshow(grad_CAM, cmap='jet', alpha=0.5)
    plt.show()


def label_list():
    # predicted label for figure title
    if not os.path.isfile(cfg_class_dir):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        request.urlretrieve(url, cfg_class_dir)

    with open(cfg_class_dir) as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


def resize_gradcam(weighted_sum, img_size):
    grad_CAM = weighted_sum

    grad_CAM = grad_CAM.unsqueeze(0)
    grad_CAM = grad_CAM.unsqueeze(0)

    upscale_layer = torch.nn.Upsample(
        scale_factor=img_size/grad_CAM.shape[-1], mode='bilinear')
    relu_layer = torch.nn.ReLU()

    grad_CAM = upscale_layer(grad_CAM)
    grad_CAM = relu_layer(grad_CAM.squeeze())
    grad_CAM = grad_CAM.squeeze().detach().numpy()
    return grad_CAM


def vis_gradcam(model, img, save_feat, save_grad):
    # during foward propagation, saving output value of target layer
    def hook_feat(module, input, output):
        save_feat.append(output)
        return output

    # dufring backward propagation, saving grad value of target layer
    def hook_grad(grad):
        save_grad.append(grad)
        return grad

    model.eval()

    # defining target layer
    target_layer = getattr(model, cfg_layer_name)
    target_layer = target_layer[cfg_layer_idx]
    target_layer = getattr(target_layer, cfg_operator)
    target_layer.register_forward_hook(hook_feat)

    output = model(img)

    save_feat[0].register_hook(hook_grad)

    output = output[0]
    loss = output[torch.argmax(output)]
    loss.backward()

    # for Grad-CAM
    gap_layer = torch.nn.AdaptiveAvgPool2d(1)
    alpha = gap_layer(save_grad[0].squeeze())
    A = save_feat[0].squeeze()
    weighted_sum = sum(alpha * A)
    grad_CAM = resize_gradcam(weighted_sum, img.shape[-1])

    # for plotting img
    img_np = image_tensor_to_numpy(img)
    img_np = img_np[0]
    img_np = normalize(img_np)

    # label of img
    labels = label_list()
    _, index = torch.max(output, -1)
    label = labels[index]

    show_image(label, img_np, grad_CAM)


def main():
    # dataset for plotting
    transform = getattr(import_module("dataset"), cfg_transform)
    test_transform = transform(cfg_resize, cfg_mean, cfg_mean)
    dataset = getattr(import_module("dataset"), cfg_dataset)
    test_dataset = dataset(cfg_data_root, transform=test_transform)

    model = getattr(models, cfg_model)
    model = model(pretrained=True)

    # gradcam of images in ./data/
    for img in test_dataset:
        save_feat = []
        save_grad = []
        vis_gradcam(model, img, save_feat, save_grad)


if __name__ == '__main__':
    main()
