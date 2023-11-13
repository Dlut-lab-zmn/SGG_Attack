# Copyright (c) Mn,Zhao. All Rights Reserved.
import os
import torch
import bisect
import copy
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data import datasets as D
from maskrcnn_benchmark.data import samplers
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.miscellaneous import save_labels
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.data.datasets.utils.config_args import config_tsv_dataset_args
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.collate_batch import BatchCollator, BBoxAugCollator
from maskrcnn_benchmark.data import transforms as T
import cv2
import numpy as np
def recover_orimg(images, iter, normal,save_path, save = False):
    img1 = (images.squeeze()*normal[1] + normal[0])/255  
    ori_img = np.array(img1[[2,1,0]].permute((1,2,0)).contiguous()*255)
    if save:
        cv2.imwrite(os.path.join(save_path, str(iter) + '.png'), ori_img)
    return ori_img
def build_dataset(cfg, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        cfg: config file.
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST


    factory_list = cfg.DATASETS.FACTORY_TRAIN if is_train else cfg.DATASETS.FACTORY_TEST
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list))
    if not isinstance(factory_list, (list, tuple)):
        raise RuntimeError(
                "factory_list should be a list of strings, got {}".format(factory_list))

    datasets = []
    for i, dataset_name in enumerate(dataset_list):
        # added support for yaml input format of tsv datasets.
        if dataset_name.endswith('.yaml'):
            factory_name = factory_list[i] if i < len(factory_list) else None
            args, tsv_dataset_name = config_tsv_dataset_args(
                cfg, dataset_name, factory_name, is_train
            )

            factory = getattr(D, tsv_dataset_name)

        else:
            data = dataset_catalog.get(dataset_name)
            factory = getattr(D, data["factory"])
            args = data["args"]
            # for COCODataset, we want to remove images without annotations
            # during training
            if data["factory"] == "COCODataset":
                args["remove_images_without_annotations"] = is_train
            if data["factory"] == "PascalVOCDataset":
                args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    #color_jitter = T.ColorJitter(
    #    brightness=brightness,
    #    contrast=contrast,
    #    saturation=saturation,
    #    hue=hue,
    #)

    transform = T.Compose(
        [
            #color_jitter,
            T.Resize(min_size, max_size),
            #T.RandomHorizontalFlip(flip_horizontal_prob),
            #T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, is_for_period=False):#
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0


    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train or is_for_period)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    for dataset in datasets:

        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = 0#cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


def load(cfg):

    output_folders = [None] * len(cfg.DATASETS.TEST)#
    dataset_names = cfg.DATASETS.TEST#
    if cfg.OUTPUT_DIR:#
        for idx, dataset_name in enumerate(dataset_names):#
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)#
            mkdir(output_folder)#
            output_folders[idx] = output_folder#
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=cfg.distributed) #

    labelmap_file = config_dataset_file(cfg.DATA_DIR, cfg.DATASETS.LABELMAP_FILE)#
    return output_folders[0], dataset_names[0], data_loaders_val[0], labelmap_file

def load_data(cfg):
    return load(cfg)
