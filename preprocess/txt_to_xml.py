import os
import cv2
import glob
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom

# Label map
LABELS = {0: "WBC", 1: "RBC", 2: "Platelets"}

SPLITS = ["Training", "Testing", "Validation"]


def yolo_to_voc(x_center, y_center, width, height, img_w, img_h):
    xmin = int((x_center - width / 2) * img_w)
    ymin = int((y_center - height / 2) * img_h)
    xmax = int((x_center + width / 2) * img_w)
    ymax = int((y_center + height / 2) * img_h)
    return xmin, ymin, xmax, ymax


def convert(txt_path, img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read image: {img_path}, skipping.")
        return

    img_h, img_w, img_d = img.shape
    basename = os.path.splitext(os.path.basename(img_path))[0]
    jpg_name = basename + ".jpg"

    # Build XML tree
    annotation = Element("annotation")

    SubElement(annotation, "folder").text = "JPEGImages"
    SubElement(annotation, "filename").text = jpg_name
    SubElement(annotation, "path").text = f"/home/pi/detection_dataset/JPEGImages/{jpg_name}"

    source = SubElement(annotation, "source")
    SubElement(source, "database").text = "Unknown"

    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(img_w)
    SubElement(size, "height").text = str(img_h)
    SubElement(size, "depth").text = str(img_d)

    SubElement(annotation, "segmented").text = "0"

    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        xmin, ymin, xmax, ymax = yolo_to_voc(xc, yc, w, h, img_w, img_h)

        obj = SubElement(annotation, "object")
        SubElement(obj, "name").text = LABELS.get(class_id, str(class_id))
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"

        bndbox = SubElement(obj, "bndbox")
        SubElement(bndbox, "xmin").text = str(xmin)
        SubElement(bndbox, "ymin").text = str(ymin)
        SubElement(bndbox, "xmax").text = str(xmax)
        SubElement(bndbox, "ymax").text = str(ymax)

    # Pretty-print with closing </annotations> fixed
    raw = minidom.parseString(
        __import__("xml.etree.ElementTree", fromlist=["tostring"]).tostring(annotation, encoding="unicode")
    ).toprettyxml(indent="\t")

    # Remove the <?xml ...?> declaration line and write clean file
    lines_out = raw.split("\n")
    clean = "\n".join(lines_out[1:])  # strip <?xml version...?>

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(clean)

    print(f"  [OK] {out_path}  ({len(lines)} objects, {img_w}x{img_h})")


def main(dataset_root="."):
    total = 0
    for split in SPLITS:
        ann_dir = os.path.join(dataset_root, split, "Annotations")
        img_dir = os.path.join(dataset_root, split, "Images")

        txt_files = glob.glob(os.path.join(ann_dir, "*.txt"))
        if not txt_files:
            print(f"[{split}] No .txt files found in {ann_dir}, skipping.")
            continue

        print(f"\n[{split}] Processing {len(txt_files)} annotation(s)...")

        for txt_path in txt_files:
            basename = os.path.splitext(os.path.basename(txt_path))[0]

            # Support .jpg or .png
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = os.path.join(img_dir, basename + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  [WARN] No matching image for {txt_path}, skipping.")
                continue

            out_path = os.path.join(ann_dir, basename + ".xml")
            convert(txt_path, img_path, out_path)
            total += 1

    print(f"\nDone. Converted {total} file(s).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to Pascal VOC XML")
    parser.add_argument(
        "--dataset", default=".",
        help="Path to dataset root (contains Training/, Testing/, Validation/). Default: current directory"
    )
    args = parser.parse_args()
    main(args.dataset)
