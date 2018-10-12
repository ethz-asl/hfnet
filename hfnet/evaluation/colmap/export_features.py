import numpy as np
import os

def export_features_from_npz(path_file, out_path):
    frame1 = np.load(path_file)

    filename = os.path.splitext(os.path.basename(path_file))[0]
    out_path_and_name = os.path.join(out_path, filename) + '.jpg.txt'
    print out_path_and_name
    outfile = open(out_path_and_name, "w+")

    current_width = frame1['image_size'][0]
    original_width = 1600
    scaling = float(original_width) / current_width

    SIFT_SIZE = 128
    kp1 = frame1['keypoints'] * scaling
    outfile.write(str(kp1.shape[0]) + ' ' + str(SIFT_SIZE) + '\n')

    for keypoint in kp1:
        outfile.write(str(keypoint[0]) + ' ' + str(keypoint[1]) + ' 1 1 ')
        for x in range(0, SIFT_SIZE):
            outfile.write(str(x) + ' ')
        outfile.write('\n')

    outfile.close()

def export_feature_detections():
    out_path = 'images_upright/db'
    in_path = 'db'
    for filename in os.listdir(in_path):
        if filename.endswith(".npz"):
             export_features_from_npz(os.path.join(in_path, filename), out_path)

if __name__ == "__main__":
    export_feature_detections()
