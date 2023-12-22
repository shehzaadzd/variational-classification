#!/bin/bash

python VC.py -a resnet50 path_to_tiny_imagenet --vc  --ol --l1 0.01 --l2 0.01 --disc_layers 1 --desc idk
