import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# specify paths
paths = {
    'train_x': "/path/to/train/images",
    'train_y': "/path/to/train/annotations",
    'test_x': "/path/to/test/images",
    'test_y': "/path/to/test/annotations",
}

background_color = 0  # background color


# augmentation methods
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ShiftScaleRotate(p=0.5, interpolation=cv2.INTER_NEAREST),
        A.ShiftScaleRotate(p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=background_color),
    ], p=0.1),

    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.25),

    A.OneOf([
        A.Blur(p=0.5),
        A.GaussNoise(p=0.5),
    ], p=0.25),

    A.OneOf([
        A.MotionBlur(p=0.5),
        A.MedianBlur(p=0.5),
    ], p=0.25),

    A.CoarseDropout(p=0.1, mask_fill_value=background_color, max_height=16, max_width=16, min_height=4, min_width=4, max_holes=3),
])


# augmentation function
def augment(x_path, y_path, resize=(256,256), transforms=None, augmentation_amount=10, mode='train'):
    # resize the images and apply augmentations a certain amount of times per image
    if mode == 'train':
        for x_file in tqdm(os.listdir(x_path)):
            x = cv2.imread(os.path.join(x_path, x_file))
            x = cv2.resize(x, resize)
            y = cv2.imread(os.path.join(y_path, x_file))
            y = cv2.resize(y, resize)
            # save original
            cv2.imwrite(os.path.join(x_path, f'{x_file[:-4]}_0.png'), x)
            cv2.imwrite(os.path.join(y_path, f'{x_file[:-4]}_0.png'), y)
            for i in range(augmentation_amount):
                augmented = transforms(image=x, mask=y)
                x_aug = augmented['image']
                y_aug = augmented['mask']
                cv2.imwrite(os.path.join(x_path, f'{x_file[:-4]}_{i+1}.png'), x_aug)
                cv2.imwrite(os.path.join(y_path, f'{x_file[:-4]}_{i+1}.png'), y_aug)
    # test mode only resizes
    elif mode == 'test':
        for x_file in tqdm(os.listdir(x_path)):
            x = cv2.imread(os.path.join(x_path, x_file))
            x = cv2.resize(x, resize)
            y = cv2.imread(os.path.join(y_path, x_file))
            y = cv2.resize(y, resize)
            cv2.imwrite(os.path.join(x_path, f'{x_file[:-4]}_0.png'), x)
            cv2.imwrite(os.path.join(y_path, f'{x_file[:-4]}_0.png'), y)

# apply augmentations
augment(paths['train_x'], paths['train_y'], transforms=transforms)
augment(paths['test_x'], paths['test_y'], transforms=transforms, mode='test')
