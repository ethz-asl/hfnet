import argparse
import numpy as np
from os import path
import struct

from internal import db_handling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sift_feature_dir', required=True)
    parser.add_argument('--query_txt_file',  required=True)
    parser.add_argument('--database_file',  required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    db = db_handling.COLMAPDatabase.connect(args.database_file)
    db.create_tables()

    with open(args.query_txt_file) as f:
        for line in f:
            name, _, h, w, fx, fy, cx, cy = line.split(' ')

            params = np.array([float(fx), float(fy), float(cx), float(cy)])
            camera_id = db.add_camera(1, int(h), int(w), params)
            image_id = db.add_image(path.join('images', name), camera_id)

            featurefile = path.join(args.sift_feature_dir,
                                    path.splitext(name)[0] + '.sift')
            with open(featurefile, 'rb') as f:
                data = f.read()

            header = struct.unpack_from('iiiii', data, 0)
            _, _, num_points, num_entries, desc_size = header
            assert num_entries == 5 and desc_size == 128
            offset = 20

            keypoints = np.zeros((num_points, 2))
            for i in range(num_points):
                point = struct.unpack_from('fffff', data, offset)
                offset += 20
                keypoints[i, :] = np.array((point[1], point[0]))

            descriptors = np.zeros((num_points, desc_size))
            for i in range(num_points):
                descriptor = struct.unpack_from('128B', data, offset)
                offset += desc_size
                descriptors[i, :] = np.asarray(descriptor)

            db.add_keypoints(image_id, keypoints)
            db.add_descriptors(image_id, descriptors)

    db.commit()


if __name__ == '__main__':
    main()
