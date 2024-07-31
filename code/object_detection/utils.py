from collections import defaultdict
import os
import cv2

# bounding box 정렬 함수
def sort_by_y_x(boxes):
    groups = defaultdict(list)
    for item in boxes:
        y_min = item[1][1]
        found_group = False
        for key in groups:
            if abs(key - y_min) <= 100:
                groups[key].append(item)
                found_group = True
                break
        if not found_group:
            groups[y_min].append(item)

    for key in groups:
        groups[key].sort(key=lambda x: x[1][0])  # Sort by x_cord

    sorted_data = []
    for key in sorted(groups):  
        sorted_data.extend(groups[key])
    
    return sorted_data

# textbox/image bounding box 크롭 + 이미지 저장 함수
def save_cropped_image(image_path, save_dir, class_id, cords,idx):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = map(int, cords)
    cropped_image = image[y1:y2, x1:x2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parts = os.path.normpath(image_path).split(os.sep)
    base_filename = f"{parts[-2]}_{os.path.splitext(parts[-1])[0]}"
    filename = f"{class_id}_{base_filename}_{idx}.jpg"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, cropped_image)