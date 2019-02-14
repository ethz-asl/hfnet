import numpy as np
import os
import struct
import argparse

from internal import db_handling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_num", type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    db = db_handling.COLMAPDatabase.connect('cmu_query_slice' + str(args.slice_num) + '.db')
    db.create_tables()

    # query/img_00769_c0_1311874744514404us_rect.jpg PINHOLE 1024 768 868.99 866.06 525.94 420.04
    with open('slice' + str(args.slice_num) + '.queries_with_intrinsics.txt') as f:
        index = 0
        for line in f:
            line = line.split(' ')

            h = int(line[2])
            w = int(line[3])
            fx = float(line[4])
            fy = float(line[5])
            cx = float(line[6])
            cy = float(line[7])

            camera_id = db.add_camera(1, h, w, [fx, fy, cx, cy])
            image_id = db.add_image(os.path.join('images', line[0]), camera_id)

            featurefile = os.path.join('sift', os.path.splitext(line[0])[0] + '.sift')
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


            index += 1

    db.commit()

if __name__ == "__main__":
    main()
