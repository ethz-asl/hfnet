import argparse
import numpy as np
import os
import sqlite3

def update_intrinsics(db_file):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    camera_to_camera_id = dict()
    new_image_data = []

    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        camera_id = row[1]
        image_name = row[2]
        #images[image_name] = (image_id, camera_id)

        camera = image_name.split(os.sep)[1]
        if camera not in camera_to_camera_id:
            print camera, camera_id
            camera_to_camera_id[camera] = camera_id
        else:
            camera_id = camera_to_camera_id[camera]

        new_image_data.append((image_id, camera_id))
    cursor.close()

    cursor = connection.cursor()
    for data in new_image_data:
        new_params = [data[1], data[0]]
        print new_params
        cursor.execute('UPDATE images SET camera_id = ? WHERE image_id = ?;', new_params)

    cursor.close()
    connection.commit()
    connection.close()

def main():
    db_file = 'robotcar92_merge_intrinsics.db'

    update_intrinsics(db_file)


if __name__ == "__main__":
    main()
