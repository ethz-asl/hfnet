import argparse
import numpy as np
import sqlite3
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsics_database_file", required=True)
    parser.add_argument("--database_file_to_modify", required=True)
    args = parser.parse_args()
    return args


def db_update_intrinsics_entry(db_connection, camera_id, intrinsics_to_write):
    # Write intrinsics back.
    cursor = db_connection.cursor()

    # Python 3 version.
    #new_params = [intrinsics_to_write.tostring(), camera_id]

    new_params = [np.getbuffer(intrinsics_to_write), camera_id]
    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;', new_params)
    cursor.close()


def update_target_db(database_file, image_name_to_id_and_camera_id, camera_intrinsics):
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()

    cursor.execute('SELECT camera_id, name FROM images;')
    for row in cursor:
        camera_id = row[0]
        name = row[1]

        assert(name in image_name_to_id_and_camera_id)
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
    image_name_to_id_and_camera_id = dict()
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        camera_id = row[0]
        name = row[2]
        image_name_to_id_and_camera_id[name] = (image_id, camera_id)

    cursor.close()

    camera_intrinsics = dict()
    cursor = connection.cursor()
    cursor.execute('SELECT camera_id, params FROM cameras;')
    for row in cursor:
        camera_id = row[0]
        intrinsics = np.fromstring(row[1], dtype=np.double)
        camera_intrinsics[camera_id] = intrinsics
        print("Got intrinsics of camera ", camera_id, " which are ", intrinsics)

    cursor.close()
    connection.close()

    return image_name_to_id_and_camera_id, camera_intrinsics


def main():
  args = parse_args()

  print 'Reading the input DB'
  image_name_to_id_and_camera_id, camera_intrinsics = parse_input_db(args.intrinsics_database_file)

  print 'Updating the target DB'
  update_target_db(args.database_file_to_modify, image_name_to_id_and_camera_id, camera_intrinsics)

  print 'Done!'


if __name__ == "__main__":
    main()
