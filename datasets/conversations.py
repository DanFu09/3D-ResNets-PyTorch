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
    with open(annotation_path, 'rb') as f:
        annotations = pickle.load(f)
    if dataset == 'val':
        class_indices = {
            'long_val': 0,
            'medium_val': 1,
            'close_up_val': 2
        }
    elif dataset == 'test':
        class_indices = {
            'long_test': 0,
            'medium_test': 1,
            'close_up_test': 2
        }
    
    def generate_image_list(segment_start, segment_end, sample_duration):
        # regularly sample images from the segment
        sample_duration = int(sample_duration)
        increment = int((segment_end - segment_start + 1) / sample_duration)
        if increment == 0:
            # use all images, duplicate images at the end
            return ([
                '{}.jpg'.format(i)
                for i in range(segment_start, segment_end + 1)
            ] + [ '{}.jpg'.format(segment_end) for i in range(sample_duration - (segment_end - segment_start + 1)) ])[:sample_duration]
        else:
            # iterate through with increment
            return [
                '{}.jpg'.format(i)
                for i in range(segment_start, segment_end + 1, increment)
            ][:sample_duration]
    data = [
        {
            'video_path': os.path.join(video_path, str(segment[0])),
            'image_list': generate_image_list(segment[1], segment[2], sample_duration),
            'class_index': class_indices[class_name]
        }
        for class_name in class_indices
        for segment, _ in annotations[class_name]
    ]

    class_names = ['long', 'medium', 'close_up']
    
    return data, class_names


class ShotScale(data.Dataset):
    """
    Args:
        video_path (string): Directory containing videos.
        annotation_path (string): Name of annotation file.
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
