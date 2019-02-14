import numpy as np
import os
from collections import defaultdict
import sqlite3
from tqdm import tqdm

import argparse

import struct

from internal import nvm_to_colmap_helper
from internal import db_handling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_num", type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    image_idx_to_keypoints = defaultdict(list)
    image_idx_to_db_image_id = []
    image_data = []

    db = db_handling.COLMAPDatabase.connect('slice' + str(args.slice_num) + '.db')
    db.create_tables()

    camera_model = 2
    w = 1024
    h = 768

    with open('slice' + str(args.slice_num) + '.nvm') as f:
        line_num = 0
        total_num_images = 0
        total_num_points = 0

        num_images = 0
        num_points = 0

        for line in f:
          if line_num == 0:
              line_num += 1
              pass
          elif line_num == 1:
              total_num_images = int(line)
              break

        for i, line in enumerate(f):
            if i == total_num_images:
                break
            line_list = line.split(' ')
            image_data.append(line_list)

            #  <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
            params = np.array((float(line_list[1]), h/2, w/2, 0.0))
            camera_id = db.add_camera(camera_model, h, w, params)
            image_id = db.add_image(line_list[0], camera_id) #db.add_image(os.path.join('images', line_list[0]), camera_id)

            featurefile = os.path.join('sift', os.path.splitext(line_list[0])[0] + '.sift')

            data = open(featurefile, 'rb').read()

            header = struct.unpack_from('iiiii', data, 0)
            assert(header[3] == 5)
            assert(header[4] == 128)

            num_points = header[2]

            offset = 20;

            keypoints = np.zeros((num_points,2))
            for i in range(0, num_points):
                point = struct.unpack_from('fffff', data, offset)
                offset += 20
                keypoints[i,:] = np.array((point[1], point[0]))

            descriptors = np.zeros((num_points,128))
            for i in range(0, num_points):
                descriptor = struct.unpack_from('128B', data, offset)
                offset += 128
                descriptors[i,:] = np.asarray(descriptor)

            db.add_keypoints(image_id, keypoints)
            db.add_descriptors(image_id, descriptors)

        db.commit()

    print('Done.')



if __name__ == "__main__":
    main()
