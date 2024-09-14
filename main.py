import os

import cv2
import numpy as np
import segment_anything
import torch
import tyro

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = segment_anything.sam_model_registry["vit_h"](
    checkpoint="assets/sam_vit_h_4b8939.pth"
)
sam.to(device)

predictor = segment_anything.SamPredictor(sam)


def get_object_selection(
    image: np.ndarray, window_name: str = "Select Object"
) -> tuple[np.ndarray, np.ndarray]:
    """Shows the image and allows the user to select the object by clicking on it.

    Args:
        image (np.ndarray): The image to be shown.
        window_name (str, optional): The name of the window. Defaults to "Select Object".
    Returns:
        tuple[np.ndarray, np.ndarray]: The coordinates of the clicked pixels and their labels.
    """

    image_copy = image.copy()
    cv2.imshow(window_name, image_copy)

    clicked_pixels = []
    labels = []

    def click_event(event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixels.append([x, y])
            labels.append(1)
            cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image_copy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked_pixels.append([x, y])
            labels.append(0)
            cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, image_copy)

    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.asarray(clicked_pixels), np.asarray(labels)


def predict_object_mask(
    image: np.ndarray, clicked_pixels: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Predicts the mask of the object based on the clicked pixels.

    Args:
        image (np.ndarray): The image.
        clicked_pixels (np.ndarray): List of clicked pixel coordinates.
        labels (np.ndarray): List of labels.
    Returns:
        np.ndarray: The predicted mask.
    """

    input_points = clicked_pixels
    input_labels = labels

    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    assert len(masks) == 1, "Expected only one mask."

    return masks[0]


def get_mask_decision(image: np.ndarray, mask: np.ndarray) -> bool:
    """Shows the mask on the image and asks the user to confirm or reject it.

    Args:
        image (np.ndarray): The image.
        mask (np.ndarray): The mask.
    Returns:
        bool: True if the mask is accepted, False otherwise.
    """

    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [0, 0, 255]

    alpha = 0.5
    blended_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    cv2.imshow("Mask", blended_image)
    key = cv2.waitKey(0)
    if key == ord("y"):
        return True
    elif key == ord("n"):
        return False
    else:
        print("Invalid input, please press 'y' or 'n'.")
        return get_mask_decision()


def annotate_image(image: np.ndarray) -> np.ndarray:
    """Finds the object in the image and returns the mask.

    Args:
        image (np.ndarray): The image.
    Returns:
        np.ndarray: The mask of the object.
    """

    finished = False
    while not finished:
        clicked_pixels, labels = get_object_selection(image)
        mask = predict_object_mask(image, clicked_pixels, labels)
        finished = get_mask_decision(image, mask)
        cv2.destroyAllWindows()

    return mask


def main(path: str, extension: str, test_interval: int = 8):
    assert os.path.exists(path), f"Dir not found: {path}"
    assert os.path.isdir(path), f"Path is not a directory: {path}"

    image_names = [f for f in os.listdir(path) if f.endswith(extension)]
    image_names.sort()

    test_image_names = [
        name for idx, name in enumerate(image_names) if idx % test_interval == 0
    ]

    for image_name in test_image_names:
        mask_path = os.path.join(path, f"{image_name.split('.')[0]}_mask.npy")
        if os.path.exists(mask_path):
            continue

        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        change_mask = annotate_image(image)
        np.save(mask_path, change_mask)


if __name__ == "__main__":
    tyro.cli(main)
