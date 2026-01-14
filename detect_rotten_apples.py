import os
import argparse
import cv2
import numpy as np

#  Core helpers
def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def read_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def bgr2gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def bgr2hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

#  Pensum: Thresholding (segmentering)
def apple_mask_from_hsv_s(bgr):
    hsv = bgr2hsv(bgr)
    _, S, _ = cv2.split(hsv)
    _, mask = cv2.threshold(S, 25, 255, cv2.THRESH_BINARY)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


#  Pensum: Contours + filtering
def find_contours(bin_mask):
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_mask(shape, c):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
    return m

def contour_feats(c):
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect = w / max(h, 1)
    return area, (x, y, w, h), aspect

def pick_two_apples(contours, img_area):
    min_area = img_area * 0.001
    cand = []
    for c in contours:
        area, bbox, aspect = contour_feats(c)
        if area < min_area: 
            continue
        if aspect < 0.35 or aspect > 2.8:
            continue
        cand.append((c, area, bbox))
    cand.sort(key=lambda t: t[1], reverse=True)
    cand = cand[:2]
    cand.sort(key=lambda t: t[2][0])  # venstre->højre
    return cand

#  Production: Defekter (HSV + gradient)
def inner_mask(m, k=15):
    return cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)

def rot_mask_hsv(bgr, apple_mask):
    hsv = bgr2hsv(bgr)
    inner = inner_mask(apple_mask, 15)
    brown = cv2.inRange(hsv, (8, 55, 20), (38, 255, 150))
    rot = cv2.bitwise_and(brown, brown, mask=inner)
    return cv2.morphologyEx(rot, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

def texture_mask_gradient(bgr, apple_mask):
    gray = bgr2gray(bgr)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inner = inner_mask(apple_mask, 15)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    _, tex = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
    tex = cv2.bitwise_and(tex, tex, mask=inner)
    return cv2.morphologyEx(tex, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

def classify(apple_mask, rot, tex):
    a = int(np.count_nonzero(apple_mask))
    if a < 1000:
        return "USIKKER", 0.0, 0.0
    rr = int(np.count_nonzero(rot)) / a
    tr = int(np.count_nonzero(tex)) / a
    if tr > 0.14: return "RÅDDENT", rr, tr
    if tr > 0.09 and rr > 0.012: return "RÅDDENT", rr, tr
    return "FRISK", rr, tr

def relative_rule(info):
    if len(info) != 2:
        return
    s0 = info[0]["rot_ratio"] + info[0]["tex_ratio"]
    s1 = info[1]["rot_ratio"] + info[1]["tex_ratio"]
    r = 0 if s0 >= s1 else 1
    f = 1 - r
    info[r]["label"] = "RÅDDENT"
    info[f]["label"] = "FRISK"

#  Pensum demo: Thresholds + Edges + Blobs (analysis)
def threshold_compare(gray):
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    th_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 17)
    th_gauss = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"gray_blur": blur, "adaptive_mean": th_mean, "adaptive_gauss": th_gauss, "otsu": th_otsu}

def edges_compare(gray):
    g = cv2.GaussianBlur(gray, (13, 13), 0)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
    sob = cv2.convertScaleAbs(np.sqrt(sx*sx + sy*sy))
    canny = cv2.Canny(g, 100, 200)
    lap = cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_64F))
    return {"sobelxy": sob, "canny": canny, "laplace": lap}

def defect_spots_mask(bgr, apple_mask):
    gray = bgr2gray(bgr)
    gauss = cv2.GaussianBlur(gray, (13, 13), 0)
    bilat = cv2.bilateralFilter(gauss, 17, 75, 75)
    inner = cv2.erode(apple_mask, np.ones((9, 9), np.uint8), iterations=1)
    grad = cv2.morphologyEx(bilat, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    _, d = cv2.threshold(grad, 18, 255, cv2.THRESH_BINARY)
    d = cv2.bitwise_and(d, d, mask=inner)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

def blob_keypoints(bin_img):
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea, p.minArea, p.maxArea = True, 10, 20000
    p.filterByColor, p.blobColor = True, 255
    det = cv2.SimpleBlobDetector_create(p)
    return det.detect(bin_img)

def save_debug(debug_dir, base, imgs):
    ensure_dir(debug_dir)
    for name, img in imgs.items():
        cv2.imwrite(os.path.join(debug_dir, f"{base}_{name}.png"), img)

#  Draw result
def draw_overlay(bgr, info):
    out = bgr.copy()
    red = np.zeros_like(out); red[:, :, 2] = 255
    blue = np.zeros_like(out); blue[:, :, 0] = 255

    for a in info:
        out = cv2.addWeighted(out, 1.0, cv2.bitwise_and(red, red, mask=a["rot"]), 0.45, 0)
        out = cv2.addWeighted(out, 1.0, cv2.bitwise_and(blue, blue, mask=a["tex"]), 0.35, 0)

    for i, a in enumerate(info, 1):
        x, y, w, h = a["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        txt = f"Aeble {i}: {a['label']} | rot={a['rot_ratio']:.3f} | tex={a['tex_ratio']:.3f}"
        cv2.putText(out, txt, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out

#  Main
def process_image(path, save_dir, debug_dir, show, analysis):
    bgr = read_bgr(path)
    if bgr is None:
        print("Kunne ikke læse:", path); return

    mask = apple_mask_from_hsv_s(bgr)
    h, w = mask.shape
    apples = pick_two_apples(find_contours(mask), h * w)

    info = []
    for c, area, bbox in apples:
        am = contour_mask(mask.shape, c)
        rot = rot_mask_hsv(bgr, am)
        tex = texture_mask_gradient(bgr, am)
        label, rr, tr = classify(am, rot, tex)
        info.append({"bbox": bbox, "label": label, "rot_ratio": rr, "tex_ratio": tr, "rot": rot, "tex": tex})

    relative_rule(info)
    out = draw_overlay(bgr, info)
    print(f"{os.path.basename(path)} -> {[a['label'] for a in info]}")

    ensure_dir(save_dir)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(path)), out)

    if analysis:
        base = os.path.splitext(os.path.basename(path))[0]
        g = bgr2gray(bgr)
        th = threshold_compare(g)
        ed = edges_compare(g)

        cs = find_contours(mask)
        canv = bgr.copy()
        cv2.drawContours(canv, cs, -1, (0, 255, 0), 2, cv2.LINE_AA)

        defect = defect_spots_mask(bgr, mask)
        kps = blob_keypoints(defect)
        blobs = cv2.drawKeypoints(cv2.cvtColor(defect, cv2.COLOR_GRAY2BGR), kps, np.array([]), (0, 0, 255),
                                  cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

        save_debug(debug_dir, base, {**th, **ed, "contours": canv, "defect_mask": defect, "blobs": blobs})

    if show:
        cv2.imshow("result", out)
        cv2.imshow("apple_mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(description="Detektion af rådne æbler (lærer-stil + portefølje)")
    ap.add_argument("--input", default="images", help="Fil eller mappe med billeder (default: images)")
    ap.add_argument("--save", default="output")
    ap.add_argument("--debug", default="debug")
    ap.add_argument("--show", action="store_true", help="Vis billeder på skærmen")
    ap.add_argument("--analysis", action="store_true")
    args = ap.parse_args()

    show = args.show

    if os.path.isdir(args.input):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        )

        if not files:
            print("Ingen billeder fundet i:", args.input)
            return

        for p in files:
            process_image(p, args.save, args.debug, show, args.analysis)
        return  # <- vigtig: stop her, så vi ikke prøver at læse mappen som billede

    # ellers: single image
    process_image(args.input, args.save, args.debug, show, args.analysis)

if __name__ == "__main__":
    main()
