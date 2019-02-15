import argparse
import numpy as np
import sqlite3
import os

from internal.db_handling import array_to_blob, blob_to_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--nvm_file', required=True)
    args = parser.parse_args()
    return args


def db_image_name_dict(connection):
    cursor = connection.cursor()
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    name_to_ids = {name: (im_id, cam_id) for im_id, cam_id, name in cursor}
    cursor.close()
    return name_to_ids


def update_db_intrinsics(db_connection, camera_id, focal_length, radial_dist):
    cursor = db_connection.cursor()
    cursor.execute('SELECT params FROM cameras WHERE camera_id=?;',
                   (camera_id,))
    data = cursor.fetchall()
    assert len(data) == 1
    intrinsics = blob_to_array(data[0][0], np.double)
    cursor.close()

    # Update intrinsic values.
    intrinsics[0] = focal_length
    intrinsics[3] = radial_dist

    # Write intrinsics back.
    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;',
                   [array_to_blob(intrinsics), camera_id])
    cursor.close()


def process(nvm_data, image_name_to_id_and_camera_id, db_connection):
    name = nvm_data[0]
    focal_length = float(nvm_data[1])
    radial_dist = -float(nvm_data[9])

    name = os.path.splitext(nvm_data[0])[0] + '.jpg'
    if name.startswith('./'):
        name = name[2:]

    image_id, camera_id = image_name_to_id_and_camera_id[name]
    update_db_intrinsics(db_connection, camera_id, focal_length, radial_dist)


def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_file)
    image_name_to_id_and_camera_id = db_image_name_dict(connection)

    with open(args.nvm_file) as f:
        line = f.readline()
        while line == '\n' or line.startswith('NVM_V3'):
            line = f.readline()
        num_images = int(line)

        i = 0
        while i < num_images:
            line = f.readline()
            if line == '\n':
                continue
            data = line.split(' ')
            process(data, image_name_to_id_and_camera_id, connection)

    print('Done parsing data for', num_images, 'images.')
    connection.commit()
    connection.close()


if __name__ == '__main__':
    main()
