import argparse
import numpy as np
from os import path
import struct

from internal import db_handling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sift_feature_dir', required=True)
    parser.add_argument('--nvm_file', required=True)
    parser.add_argument('--database_file', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    db = db_handling.COLMAPDatabase.connect(args.database_file)
    db.create_tables()

    camera_model = 2
    w = 1024
    h = 768

    with open(args.nvm_file) as f:
        f.readline()
        total_num_images = int(f.readline())
        for _, line in zip(range(total_num_images), f):
            line = line.split(' ')
            name, focal, dist = line[0], line[1], line[9]

            #  <Camera> = <name> <focal> <quat WXYZ> <translation> <dist> 0
            params = np.array([float(focal), h/2, w/2, float(dist)])
            camera_id = db.add_camera(camera_model, h, w, params)
            image_id = db.add_image(name, camera_id)

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
