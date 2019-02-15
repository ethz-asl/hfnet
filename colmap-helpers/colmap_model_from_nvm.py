import argparse
import numpy as np
import os
import sqlite3

from internal.nvm_to_colmap_helper import convert_nvm_pose_to_colmap_p
from internal.db_handling import blob_to_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--nvm_file', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    return args


def db_image_name_dict(db):
    db.execute('SELECT image_id, camera_id, name FROM images;')
    return {name: (im_id, cam_id) for im_id, cam_id, name in db}


def export_cameras(db, output_dir):
    # In db file:
    # model INTEGER NOT NULL,
    # width INTEGER NOT NULL,
    # height INTEGER NOT NULL,
    # params BLOB,
    # prior_focal_length INTEGER NOT NULL)

    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]

    db.execute('SELECT camera_id, model, width, height, params FROM cameras;')
    outfile = open(os.path.join(output_dir, 'cameras.txt'), 'w')
    for camera_id, model, width, height, params in db:
        if model != 2:  # Wrong model, skip this camera.
            continue
        model = 'SIMPLE_RADIAL'
        params = blob_to_array(params, np.double)
        out = [camera_id, model, width, height] + params.tolist()
        outfile.write(' '.join(map(str, out)) + '\n')
    outfile.close()


def convert_image_data(nvm_data, name_to_image_id):
    # colmap format
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    # nvm format
    # <Name> <focal> <quaternion WXYZ> <camera center> <radial distortion> 0
    name, focal, q1, q2, q3, q4, p1, p2, p3, dist, _ = nvm_data

    if name.endswith('.png'):
        name = name[:-3] + 'jpg'
    if name.startswith('./'):
        name = name[2:]
    image_id, camera_id = name_to_image_id[name]
    assert image_id > 0

    q_nvm = np.array(list(map(float, [q1, q2, q3, q4])))
    p_nvm = np.array(list(map(float, [p1, p2, p3])))
    p1, p2, p3 = convert_nvm_pose_to_colmap_p(q_nvm, p_nvm).tolist()

    # Write the corresponding camera id from the database file, as some
    # intrinsics could have been merged in the database (and not in NVM).
    return [image_id, q1, q2, q3, q4, p1, p2, p3, camera_id, name]


def main():
    args = parse_args()

    print('Reading DB')
    connection = sqlite3.connect(args.database_file)
    db = connection.cursor()
    name_to_image_id = db_image_name_dict(db)

    print('Exporting cameras')
    export_cameras(db, args.output_dir)

    print('Creating empty Points3D file')
    open(os.path.join(args.output_dir, 'points3D.txt'), 'w+').close()

    print('Reading NVM')
    with open(args.nvm_file) as f:
        line = f.readline()
        while line == '\n' or line.startswith('NVM_V3'):
            line = f.readline()
        total_num_images = int(line)
        print('Num images', total_num_images)

        outfile = open(os.path.join(args.output_dir, 'images.txt'), 'w')
        i = 0
        while i < total_num_images:
            line = f.readline()
            if line == '\n':
                continue
            data = convert_image_data(line.split(), name_to_image_id)
            outfile.write(' '.join(map(str, data)) + '\n')
            i += 1
        outfile.close()

    print('Done')
    db.close()
    connection.close()


if __name__ == '__main__':
    main()
