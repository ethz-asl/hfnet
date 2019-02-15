import numpy as np
import os
from collections import defaultdict
import sqlite3
from tqdm import tqdm
import argparse

from internal.nvm_to_colmap_helper import convert_nvm_pose_to_colmap_p
from internal.db_handling import blob_to_array


# NVM format
# <Camera> = <Name> <focal> <quaternion WXYZ> <camera center> <distortion> 0
# <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
# <Measurement> = <Image index> <Feature Index> <xy>

# Colmap format
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--nvm_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    return args


def image_id_from_name(image_name, db):
    db.execute('SELECT image_id FROM images WHERE name=?;', (image_name,))
    data = db.fetchall()
    assert len(data) == 1, image_name
    return data[0][0]


def camera_params_from_image_id(image_id, db):
    db.execute('SELECT camera_id FROM images WHERE image_id=?;', (image_id,))
    data = db.fetchall()
    assert len(data) == 1, image_id
    camera_id = data[0][0]

    db.execute('SELECT width, height, params FROM cameras WHERE camera_id=?;',
               (camera_id,))
    data = db.fetchall()
    assert len(data) == 1, image_id
    width, height, params = data[0]
    return camera_id, width, height, blob_to_array(params, np.double)


def read_keypoints_from_db(image_id, db):
    db.execute('SELECT data FROM keypoints WHERE image_id=?;', (image_id,))
    data = db.fetchall()
    assert len(data) == 1, image_id
    return blob_to_array(data[0][0], np.float32, shape=(-1, 2))


def export_image_data(image_data, image_idx_to_keypoints,
                      image_idx_to_db_image_id, db, output_dir):
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    out_images = open(os.path.join(output_dir, 'images.txt'), 'w')
    out_cameras = open(os.path.join(output_dir, 'cameras.txt'), 'w')

    for image_idx, nvm_data in tqdm(enumerate(image_data), unit='images'):
        name, focal, q1, q2, q3, q4, p1, p2, p3, dist, _ = nvm_data

        q_nvm = np.array(list(map(float, [q1, q2, q3, q4])))
        p_nvm = np.array(list(map(float, [p1, p2, p3])))
        p1, p2, p3 = convert_nvm_pose_to_colmap_p(q_nvm, p_nvm).tolist()

        db_image_id = image_idx_to_db_image_id[image_idx]
        camera_id, width, height, params = camera_params_from_image_id(
            db_image_id, db)

        out = [db_image_id, q1, q2, q3, q4, p1, p2, p3, camera_id, name]
        out_images.write(' '.join(map(str, out)) + '\n')

        # cameras.txt format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        out = [camera_id, 'SIMPLE_RADIAL', width, height, focal,
               params[1], params[2], -1*float(dist)]
        out_cameras.write(' '.join(map(str, out)) + '\n')

        # Now export the keypoints that correspond to 3D points
        keypoints = read_keypoints_from_db(db_image_id, db)
        associated_keypoints = dict()
        if image_idx in image_idx_to_keypoints:
            matching_keypoints = image_idx_to_keypoints[image_idx]
            for kp_index, _, _, point_index in matching_keypoints:
                assert kp_index not in associated_keypoints
                associated_keypoints[kp_index] = point_index

        out = ''
        for kp_index, keypoint in enumerate(keypoints):
            if kp_index in associated_keypoints:
                point_idx = associated_keypoints[kp_index]
            else:
                point_idx = -1
            out += f'{keypoint[0]} {keypoint[1]} {point_idx} '
        out_images.write(out + '\n')

    out_images.close()
    out_cameras.close()


def main():
    args = parse_args()
    connection = sqlite3.connect(args.database_file)
    db = connection.cursor()

    image_idx_to_keypoints = defaultdict(list)
    image_idx_to_db_image_id = []

    with open(args.nvm_file) as f:
        # Skip empty lines and read the number of images
        line = f.readline()
        while line == '\n' or line.startswith('NVM_V3'):
            line = f.readline()
        num_images = int(line)
        print('Num images', num_images)

        image_data = []
        i = 0
        while i < num_images:
            line = f.readline()
            if line == '\n':
                continue
            data = line.split(' ')
            image_data.append(data)
            image_idx_to_db_image_id.append(image_id_from_name(data[0], db))
            i += 1
        print('Image index list length:', len(image_idx_to_db_image_id))

        # Skip empty lines and read the number of 3D points
        line = f.readline()
        while line == '\n':
            line = f.readline()
        num_points = int(line)
        print('Will export', num_points, '3D point entries.')

        out_points = open(os.path.join(args.output_dir, 'points3D.txt'), 'w')
        pbar = tqdm(total=num_points, unit='pts')
        idx = 0
        while idx < num_points:
            line = f.readline()
            if line == '\n':
                continue

            data = line.split()
            x, y, z, r, g, b, num_observations = data[:7]
            err = 1  # fake reprojection error since no provided in NVM
            out = ' '.join(map(str, [idx, x, y, z, r, g, b, err]))
            for j in range(int(num_observations)):
                start_index = 7 + 4 * j  # Offset + 4 values per observation
                img_index, kp_index, kx, ky = data[start_index:start_index + 4]
                image_idx_to_keypoints[int(img_index)].append(
                    (int(kp_index), float(kx), float(ky), idx))
                db_image_id = image_idx_to_db_image_id[int(img_index)]
                out += f' {db_image_id} {kp_index}'

            out_points.write(out + '\n')
            idx += 1
            pbar.update(1)
        out_points.close()
        pbar.close()

    print('Points3D done. Now exporting images.')
    export_image_data(image_data, image_idx_to_keypoints,
                      image_idx_to_db_image_id, db, args.output_dir)

    print('Done.')
    db.close()
    connection.close()


if __name__ == '__main__':
    main()
