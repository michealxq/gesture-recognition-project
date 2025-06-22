# scripts/generate_crops.py

import os, cv2, tqdm, mediapipe as mp

SRC_ROOT = r"C:\Users\AliRAMADAN\Desktop\hagrid_data_split"    # where train/val/test folders live
DST_ROOT = r"C:\Users\AliRAMADAN\Desktop\hagrid_crops"         # where we'll dump crops

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def process_folder(split):
    src_split = os.path.join(SRC_ROOT, split)
    dst_split = os.path.join(DST_ROOT, split)
    for cls in os.listdir(src_split):
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(dst_split, cls)
        os.makedirs(dst_cls, exist_ok=True)
        for fn in tqdm.tqdm(os.listdir(src_cls), desc=f"{split}/{cls}"):
            src_img = os.path.join(src_cls, fn)
            dst_img = os.path.join(dst_cls, fn)
            img = cv2.imread(src_img)
            if img is None: continue
            h, w = img.shape[:2]
            res = mp_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            # compute square bbox + 50% padding
            pts = [(int(lm.x*w), int(lm.y*h))
                   for lm in res.multi_hand_landmarks[0].landmark]
            xs, ys = zip(*pts)
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            side = max(x1-x0, y1-y0)
            cx, cy = (x0+x1)//2, (y0+y1)//2
            pad = int(side * 0.5)
            half = side//2 + pad
            x0_, x1_ = max(0, cx-half), min(w, cx+half)
            y0_, y1_ = max(0, cy-half), min(h, cy+half)
            crop = img[y0_:y1_, x0_:x1_]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (224,224))
            cv2.imwrite(dst_img, crop)

for split in ["val"]:
    process_folder(split)
