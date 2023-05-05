# https://github.com/NVIDIA/DALI/blob/master/docs/examples/use_cases/pytorch/resnet50/main.py

import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    if is_training:
        images, labels = fn.readers.file(file_root=os.path.join(data_dir, 'train'),
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         random_shuffle=True,
                                         pad_last_batch=True,
                                         name="Reader")
    else:
        images, labels = fn.readers.file(file_root=os.path.join(data_dir, 'val/images'),
                                         file_list=os.path.join(data_dir, 'val/val_annotations.txt'),
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         random_shuffle=False,
                                         pad_last_batch=True,
                                         name="Reader")

    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)

    images = fn.resize(images,
                       device=dali_device,
                       size=size,
                       mode="not_smaller",
                       interp_type=types.INTERP_TRIANGULAR)

    if is_training:
        images = fn.random.flip_left_right(images)

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    labels = labels.gpu()
    return images, labels


def get_tiny_dali_data_loader(args):
    crop_size = 64
    val_size = 64

    path = 'tiny-imagenet-200/'
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.num_workers,
                                device_id=args.rank,
                                seed=12 + args.rank,
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali == 'cpu',
                                shard_id=args.rank,
                                num_shards=args.world_size,
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.num_workers,
                                device_id=args.rank,
                                seed=12 + args.rank,
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                dali_cpu=args.dali == 'cpu',
                                shard_id=args.rank,
                                num_shards=args.world_size,
                                is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    return train_loader, val_loader
