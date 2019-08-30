import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import pickle


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, image_names, image_loader):
    video = []
    for img in image_names:
        image_path = os.path.join(video_dir_path, img)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(video_path, annotation_path, dataset, sample_duration):
    """
    Args:
        video_path (string): Directory containing videos.
        annotation_path (string): Name of annotation file.
        dataset (string): "val" or "test"
        sample_duration (int): Number of frames per segment.
    
    Need to return:
        data (list): List of {video_path, image_list, class_index} dicts of (string, list, int).
        class_names (dict): Dict with items (class_name, class_index).
    """
    annotations = []
    with open(os.path.join(annotation_path, '{}.txt'.format(dataset)), 'r') as f:
        for line in f.readlines():
            video, segment, label = line.split(' ')
            annotations.append((video, segment, label))
    
    annotations_by_video = {}
    for video, segment, label in annotations:
        if video not in annotations_by_video:
            annotations_by_video[video] = []
        annotations_by_video[video].append((segment, int(label.strip())))
    
    # def split_into_segments(image_label_list):
    #     num_images = len(image_label_list)
    #     segments = []
    #     for start_idx in range(0, num_images, sample_duration):
    #         segment = image_label_list[start_idx:start_idx + sample_duration]
    #         while len(segment) < sample_duration:
    #             segment.append(segment[-1])
    #         segments.append(segment)
    #     return segments

    # segments_by_video = {
    #     video: split_into_segments(annotations_by_video[video])
    #     for video in annotations_by_video
    # }

    data = [
        {
            'video_path': os.path.join(video_path, video),
            'image_list': [
                '{:04d}_{:02d}.jpg'.format(int(segment), img)
                for img in range(sample_duration)
            ],
            'class_index': label
        }
        for video in annotations_by_video
        for segment, label in annotations_by_video[video]
    ]

    class_names = [0, 1]

    return data, class_names


class DenseSample(data.Dataset):
    """
    Args:
        video_path (string): Directory containing videos.
        annotation_path (string): Path of annotation file.
        dataset (string): "val" or "test"
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        sample_duration (int): Number of frames per segment.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_names (dict): Dict with items (class_name, class_index).
        data (list): List of {video_path, image_list, class_index} dicts of (string, list, int)
    """

    def __init__(self,
                 video_path,
                 annotation_path,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            video_path, annotation_path, dataset, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target) where target is class_index of the target class.
        """
        video_path = self.data[index]['video_path']
        image_list = self.data[index]['image_list']
        clip = self.loader(video_path, image_list)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = self.data[index]['class_index']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)
