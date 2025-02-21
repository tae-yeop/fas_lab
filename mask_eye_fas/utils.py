
def crop_eye_region(image, landmarks, indices):
    h, w = image.shape[:2]

    xs = []
    ys = []

    for idx in indices:
        lm = landmarks[idx]  # (x=, y=, z=)
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        xs.append(x_px)
        ys.append(y_px)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 약간의 마진
    margin = 20
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, w)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, h)
    
    # crop
    eye_roi = image[y_min:y_max, x_min:x_max].copy()
    return eye_roi

