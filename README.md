
# Transfer learning-based package for image retrieval.

A simple package that implements transfer learning (ResNet50) for image retrieval.

The package exposes two functions:

- A function ("preprocess_images") that pre-processes and assigns a dataset_id to a local dataset of images, and

- A function ("find_similar_images") that accepts a URL to an online image, the dataset_id of an image dataset (as assigned by the first function), and an integer K.
The function then searches through the images in the specified dataset and returns the top-K with the highest similarity to the online image.


# Installation Guide

Follow these steps to install the tranfer-learning package:

## Prerequisites

- Ensure you have Python installed on your system. This package requires Python 3.6 or later.
- [Optional] If you plan to use a virtual environment (recommended), ensure you have `virtualenv` installed. If not, you can install it using pip:

On Windows

```bash
pip install virtualenv
```

## Step 1: Set Up a New Virtual Environment (Recommended)

Creating a virtual environment for the 'transfer_learning' package helps to keep dependencies required by different projects separate.
To create a new virtual environment, follow these steps:

Navigate to your desired directory:

```bash
cd path\to\your\new_venv\directory
```

Create a new virtual environment:

```bash
python -m virtualenv name_your_venv
```

Activate the Virtual Environment:


```bash
.\venv\Scripts\activate
```

## Step 2: Install the Package

### Cloning the Repository

First, clone the repository using Git:

```bash
git clone https://github.com/yourusername/Tranfer-Learning.git
cd transfer_learning
```

### Building the package

Generate distribution packages by running these commands from the same directory where pyproject.toml is located:

```bash
python -m pip install --upgrade build
python -m build
```

#### Installing package

Navigate to the directory containing the transfer_learning distribution files (*.whl or *.tar.gz).

```bash
cd dist
```

Install the package using pip:

For wheel file (.whl):

```bash
pip install transfer_learning-0.0.1-py3-none-any.whl
```

For source distribution (.tar.gz):

```bash
pip install transfer_learning-0.0.1.tar.gz
```

## Step 3: Verify Installation

After installation, you can verify the package is installed correctly by running:

```bash
pip list
```

This command lists all installed packages in the current environment. Look for the 'transfer_learning' package in this list.

## Step 4: Using the Package

Please keep in mind that the directory structure should be organised as follows:

```bash
path/to/image_database/
├── dataset_1/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── another_dataset/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── ...
```

Once your environment is activated, open a Python interactive shell by simply typing python in your command line.

### Import Your Package and Test Its Functionalities

```python
import transfer_learning
from transfer_learning.image_preprocessing import preprocess_images
from transfer_learning.image_indexing import find_similar_images

# 1st function
main_directory_img_database = '/path/to/image_database'
processed_images = preprocess_images(main_directory_img_database)

# 2nd function
img_url = 'http://random.com/online_image.jpg'
top_K = find_similar_images(processed_images, 'dataset_1', img_url, K=5)
print(top_K)
```
