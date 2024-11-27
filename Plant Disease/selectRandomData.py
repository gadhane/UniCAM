import os
import shutil
import random

def copy_random_images(source_dir, target_dir, num_images=5):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"The source directory {source_dir} does not exist.")
        return

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over each folder in the source directory
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # Check if it's a directory
        if os.path.isdir(class_path):
            images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            # Select 5 random images
            selected_images = random.sample(images, min(len(images), num_images))

            # Create a target folder for the class
            target_class_path = os.path.join(target_dir, class_name)
            if not os.path.exists(target_class_path):
                os.makedirs(target_class_path)

            # Copy selected images to the target folder
            for img in selected_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(target_class_path, img))
if __name__ == '__main__':
    source_directory = 'Images/Data/train'  # Replace with your source directory path
    target_directory = 'Images/cam'  # Replace with your target directory path
    copy_random_images(source_directory, target_directory)
