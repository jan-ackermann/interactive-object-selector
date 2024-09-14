# Interactive Object Selector
This script serves the purpose to segment single objects in images.

To achieve this, we use SAM.

## Prerequisites

Please make sure to install 

```opencv-python, numpy, torch, segment_anything, tyro```

and download the segment_anything model that you would like to use.

## Run

Invoke the script with

```python3 main.py --path <path_to_your_images> --ext <extension_of_images>```
