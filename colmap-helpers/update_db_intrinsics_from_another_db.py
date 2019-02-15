import argparse
import numpy as np
import sqlite3

from internal.db_handling import array_to_blob, blob_to_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_database_file', required=True)
    parser.add_argument('--database_file_to_modify', required=True)
    args = parser.parse_args()
    return args


def db_update_intrinsics_entry(db_connection, camera_id, intrinsics_to_write):
    cursor = db_connection.cursor()
    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;',
                   [array_to_blob(intrinsics_to_write), camera_id])
    cursor.close()


def update_target_db(database_file, image_name_to_id_and_camera_id,
                     camera_intrinsics):
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()

    cursor.execute('SELECT camera_id, name FROM images;')
    for camera_id, name in cursor:
        assert name in image_name_to_id_and_camera_id
        src_image_id, src_camera_id = image_name_to_id_and_camera_id[name]
        assert(src_camera_id in camera_intrinsics)
        intrinsics_to_write = camera_intrinsics[src_camera_id]

        db_update_intrinsics_entry(connection, camera_id, intrinsics_to_write)

    cursor.close()
    connection.commit()
    connection.close()


def parse_input_db(database_file):
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()

    # Get a mapping between image ids and image names and camera ids.
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    name_to_ids = {name: (im_id, cam_id) for im_id, cam_id, name in cursor}

    cursor.execute('SELECT camera_id, params FROM cameras;')
    intrinsics = {cam_id: blob_to_array(p, np.double) for cam_id, p in cursor}

    cursor.close()
    connection.close()
    return name_to_ids, intrinsics


def main():
    args = parse_args()

    print('Reading the input DB')
    image_name_to_id_and_camera_id, camera_intrinsics = parse_input_db(
        args.intrinsics_database_file)

    print('Updating the target DB')
    update_target_db(
        args.database_file_to_modify, image_name_to_id_and_camera_id,
        camera_intrinsics)

    print('Done!')


if __name__ == '__main__':
    main()
