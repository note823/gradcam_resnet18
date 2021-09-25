from easydict import EasyDict
import os

__C = EasyDict()
cfg = __C

# Current directory
__C.ROOT_DIR = os.path.abspath('.')

# Directory of test picture
__C.DATA_DIR = os.path.join(__C.ROOT_DIR, 'data')

# Model
__C.MODEL = "resnet18"

# (__C.LAYER_NAME, __C.LAYER_SEQUENTIAL_IDX, __C.OPERATOR) means target layer
# Layer name
__C.LAYER_NAME = "layer4"

# Sequential idx of layer
__C.LAYER_SEQUENTIAL_IDX = 1

# Operator
__C.OPERATOR = "conv2"

# Directory of imagenet_classes.txt
# download url: # https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt
__C.IMAGET_CLASS_DIR = os.path.join(__C.ROOT_DIR, 'imagenet_classes.txt')

# Mean for transform
__C.MEAN = [0.485, 0.456, 0.406]

# Standard deviation for transform
__C.STD = [0.229, 0.224, 0.225]

# Resize for transform
__C.RESIZE = 224

# Dataset
__C.DATASET = "CustomDataset"

# Transform
__C.TRANSFORM = "CustomAugmentation"
