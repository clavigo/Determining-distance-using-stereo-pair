from pathlib import Path
import cv2

def draw_label_with_background(img, text, pos):
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=2
    font_thickness=5
    text_color=(0, 0, 0)
    bg_color=(255, 255, 255)
    x, y = pos

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    top_left = (x, y - text_height - baseline)
    bottom_right = (x + text_width, y)

    cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=cv2.FILLED)

    cv2.putText(img, text, (x, y - baseline), font, font_scale, text_color, font_thickness)


def annotate_img(image_path, objects):
    img = cv2.imread(image_path)
    filename = Path(image_path).stem

    print(objects)

    for left, right in objects:
        x1, y1, x2, y2 = left["bbox"]
        class_name = left["class"]
        distance = left["distance"]

        if distance <= 0:
            continue

        label = f"{class_name}, {distance:.1f} cm"

        draw_label_with_background(img, label, (x1, y1))

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=10)

        # cv2.putText(
        #     img,
        #     label,
        #     (x1, y1 - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=3,
        #     color=(0, 0, 255),
        #     thickness=10
        # )

    cv2.imwrite(f'results/annotated/{filename}_annotated.jpg', img)