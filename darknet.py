import torch
import torch.nn as nn
import numpy as np

import os
from util import *

cfgfile = 'cfg/yolov3.cfg'

def parse_cfg(cfgfile):
    """
    
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            # print(line, sep='\n')
            block['type'] = line[1: -1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

def create_modules(blocks):
    net_info = blocks[0] # Capture information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of block
        # create a new module for the block
        # append to module_list
        if x['type'] == 'convolutional':
            # get the info about the layer
            activation = x['activation']
            try: # has BN
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            
            if padding: 
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
    
            # add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)
            
            # add BN layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
                
            # check the activation
            # it is either Linear or Leaky ReLU for YOLO
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)
        
        # if it's an upsampling layer, we use Bilinear2dUpsampling
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.UpsamplingBilinear2d(scale_factor=stride)
            module.add_module('upsample_{0}'.format(index), upsample)

        # if it is a route layer, one or two paras
        # one para: route to one layer
        # two para: route to two layers
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # start of a route
            start = int(x['layers'][0])
            # end, if there exists one
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            # positive anotation
            if start > 0:
                start -= index
            if end > 0:
                end -= index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route) 
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
                
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            
            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)
            
        # shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        
        # do some bookkeeping
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)
            
class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors
        
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA):
        # two purposes
        # first: calculate the output
        # second: transform the output detection feature maps in a way
        #   that it can be processed easier (like normalize the dimensions)
        modules = self.blocks[1:]
        outputs = {}
        # print("QWQWQWQ")
        # print(len(self.blocks), len(self.module_list))
        # print(self.module_list)
        # types = [i['type'] for i in self.blocks]
        # for i in range(len(types)):
        #     print(types[i], sep='\n')
        # print("QWQWQWQ")
        
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = x.float()
                x = self.module_list[i](x)
                
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] -= - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= i
                    if len(layers) == 1:
                        x = outputs[i + layers[0]]
                    else:
                        if layers[1] > 0:
                            layers[1] -= i
                        map1 = outputs[i + layers[0]]
                        map2 = outputs[i + layers[1]]
                        x = torch.cat((map1, map2), 1)
                        
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
            
            elif module_type == 'yolo':
                # print("DEBUG: yolo")
                # print(i)
                anchors = self.module_list[i][0].anchors
                # get the input dimensions
                inp_dim = int(self.net_info['height'])
                # get the number of classes
                num_classes = int(module['classes'])
                # transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
            
        return detections
                
            
def get_test_input():
    img_ = cv2.imread('./imgs/dog-cycle-car.png')
    img_ = cv2.resize(img_, (416, 416))
    img_ = img_[:,:,::-1].transpose((2, 0, 1)) # BGR -> RGB
    img_ = img_[np.newaxis,:,:,:] / 255.0 # convert to float
    img_ = torch.tensor(img_, dtype=torch.float)
    return img_

print(os.getcwd())
model = Darknet('cfg/yolov3.cfg')
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print(pred)