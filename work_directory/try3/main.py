import os
import time

from yolo import detect_objects, pair_objects
from resnet import resnet
from distance import compute_depth
from labels import annotate_img

def main():
    start = time.time()

    image_width = 4000
    h_fov_deg = 67
    baseline_cm = 21

    os.makedirs("results", exist_ok=True)

    left_image = 'images/left_30cm.jpg'
    right_image = 'images/right_30cm.jpg'

    objects1 = detect_objects(left_image)
    objects2 = detect_objects(right_image)

    paired = pair_objects(objects1, objects2)

    print(paired)

    for i, (left, right) in enumerate(paired):
        print(f"\nPair {i+1}:")
        print("Left:", left["image"] if left else "None")
        print("Right:", right["image"] if right else "None")
        print(left)

    paired = resnet(paired)

    for i, (left, right) in enumerate(paired):
        print(f"\nPair {i+1}:")
        print("Left:", left["image"] if left else "None")
        print("Right:", right["image"] if right else "None")
        print(left)

    compute_depth(paired, image_width, h_fov_deg, baseline_cm)

    annotate_img(left_image, paired)

    end = time.time()
    print(f"Time: {end - start:.4f} sec")

if __name__ == "__main__":
    main()