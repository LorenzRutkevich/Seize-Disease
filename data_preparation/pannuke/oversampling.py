import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


def oversample_dataset(
    image_path,
    mask_path,
    out_image,
    out_mask,
    classes_to_duplicate,
    style="normal",
    absent_classes=[5],
    num_classes=6,
):
    """
    This function loads a dataset of images and masks, duplicates them if they contain the classes of interest,
    and saves the resulting images and masks to specified output directories.
    The output is a set of images and masks that contain the classes of interest if set to 'normal'.
    If set to 'interest only', the output will contain the classes of interest and not the classes in 'absent_classes'.
    """
    assert style in [
        "normal",
        "interest only",
    ], "Style must be either 'normal' or 'interest only'."
    # Load images and masks
    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        img = load_img(os.path.join(image_path, img_file))
        img = img_to_array(img)
        images.append(img)

        mask = load_img(os.path.join(mask_path, mask_file), color_mode="grayscale")
        mask = img_to_array(mask)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    images /= 255.0

    # Apply one-hot encoding to masks
    masks = to_categorical(masks, num_classes=num_classes)
    print("Masks shape:", masks.shape)

    # Swap values in the mask, so that the first channel corresponds to the background
    masks[:, :, :, [0, -1]] = masks[:, :, :, [-1, 0]]

    # Create output directories if they don't exist
    if not os.path.exists(out_image):
        os.makedirs(out_image)

    if not os.path.exists(out_mask):
        os.makedirs(out_mask)

    # Plot an image and its mask to check file integrity
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(images[0])
    ax[1].imshow(masks[0, :, :, 0:3])
    plt.show()
    # Loop through the masks and duplicate images and masks if they contain the classes of interest, depending on the style
    target = 0
    for i in tqdm(range(len(masks))):
        mask = masks[i]
        if style == "normal":
            if np.max(mask[:, :, classes_to_duplicate]) > 0:
                target += 1
                image_name = image_files[i]
                mask_name = mask_files[i]
                # Copy the original image and mask to the output directory
                shutil.copy(
                    os.path.join(image_path, image_name),
                    os.path.join(out_image, image_name),
                )
                shutil.copy(
                    os.path.join(mask_path, mask_name),
                    os.path.join(out_mask, mask_name),
                )
        elif style == "interest only":
            if (
                np.max(mask[:, :, classes_to_duplicate]) > 0
                and np.max(mask[:, :, absent_classes]) == 0
            ):
                target += 1
                image_name = image_files[i]
                mask_name = mask_files[i]
                # Copy the original image and mask to the output directory
                shutil.copy(
                    os.path.join(image_path, image_name),
                    os.path.join(out_image, image_name),
                )
                shutil.copy(
                    os.path.join(mask_path, mask_name),
                    os.path.join(out_mask, mask_name),
                )
    print(f"Done, found {target} images matching the criteria.")


if __name__ == "__main__":
    # Define input and output paths
    image_path = "path/to/images"
    mask_path = "path/to/sem_masks"
    out_image = "path/to/images/"
    out_mask = "path/to/masks/"

    os.makedirs(out_image, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    # Define classes to duplicate
    classes_to_duplicate = [1]

    # Run the function
    oversample_dataset(
        image_path,
        mask_path,
        out_image,
        out_mask,
        classes_to_duplicate,
        style="interest only",
        absent_classes=[3],
    )
