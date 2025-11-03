from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from data.LOLdataset import LOLv1DatasetFromFolder, LOLv2DatasetFromFolder, LOLv2SynDatasetFromFolder
from data.eval_sets import *
from data.SICE_blur_SID import *

def transform1(size=256):
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])

def transform2():
    return Compose([ToTensor()])

def transform3():
    return Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])
