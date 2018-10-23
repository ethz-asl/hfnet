import argparse
import cv2
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", required=True)
    # The images must be in the same directory as output. See:
    # https://colmap.github.io/tutorial.html#feature-detection-and-extraction
    parser.add_argument("--image_dir", required=True)
    args = parser.parse_args()
    return args


def export_features_from_npz(filename, in_path, out_path):
    path_file = os.path.join(in_path, filename)
    frame1 = np.load(path_file)

    filename = os.path.splitext(os.path.basename(path_file))[0]

    # Load a corresponding image.
    img_file = os.path.join(out_path, filename) + '.jpg'
    img = cv2.imread(img_file, 0)
    original_height, original_width = img.shape

    out_path_and_name = os.path.join(out_path, filename) + '.jpg.txt'
    print out_path_and_name
    outfile = open(out_path_and_name, "w+")

    current_width = frame1['image_size'][0]
    current_height = frame1['image_size'][1]

    scaling = float(original_width) / current_width

    SIFT_SIZE = 128
    # Scale the keypoints according to the scaling factor.
    kp1 = frame1['keypoints'] * scaling
    outfile.write(str(kp1.shape[0]) + ' ' + str(SIFT_SIZE) + '\n')

    for keypoint in kp1:
        outfile.write(str(keypoint[0]) + ' ' + str(keypoint[1]) + ' 1 1 ')
        assert (keypoint[0] <= original_width)
        # Generate some dummy SIFT values as we will anyway use external
        # from a matches.txt file.
        for x in range(0, SIFT_SIZE):
            outfile.write(str(x) + ' ')
        outfile.write('\n')

    outfile.close()


def export_feature_detections():
    args = parse_args()
    for filename in os.listdir(args.npz_dir):
        if filename.endswith(".npz"):
             export_features_from_npz(filename, args.npz_dir, args.image_dir)


if __name__ == "__main__":
    export_feature_detections()
