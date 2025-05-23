import math

def compute_depth(paired, image_width, h_fov_deg, baseline_cm):
    # result = []

    h_fov_rad = math.radians(h_fov_deg)
    f = image_width / (2 * math.tan(h_fov_rad / 2))

    for left, right in paired:
        bbox_left = left["bbox"]
        bbox_right = right["bbox"]

        x_left_center = (bbox_left[0] + bbox_left[2]) / 2
        x_right_center = (bbox_right[0] + bbox_right[2]) / 2

        disparity = x_left_center - x_right_center

        if disparity == 0:
            print('Failed to compute distance')

            left['distance'] = 0
            right['distance'] = 0

            del left['image']
            del right['image']
            
            continue
        
        Z = (f * baseline_cm) / disparity

        print("Left:", left["image"])
        print("Right:", right["image"])
        print('Distance: ',Z)

        left['distance'] = Z
        right['distance'] = Z

        del left['image']
        del right['image']

        # result.append((left, right))
    
    # for pair in paired[:]:
    #     left, right = pair
    #     if not left and not right:
    #         paired.remove(pair)

    # return pair