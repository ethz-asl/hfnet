import argparse
import numpy as np
import os
import PIL.ExifTags
import PIL.Image
import sqlite3
from collections import defaultdict

# Interesting tags:
# 271: u'NIKON', 272: u'COOLPIX S4'
# focal length 37386

def get_images_from_db(db_file):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    images = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        camera_id = row[1]
        image_name = row[2]
        images[image_name] = (image_id, camera_id)
    cursor.close()

    return images

def read_intrinsics_db(db_file, filename_list, images):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    assert(len(filename_list) > 0)

    num_horizontal = 0
    num_vertical = 0

    for filename in filename_list:
        assert(filename in images.keys())

        camera_id = images[filename][1]
        cursor.execute("SELECT params FROM cameras WHERE camera_id=?;", (camera_id,))
        rows = cursor.fetchall()
        assert(len(rows) == 1)
        intrinsics = np.fromstring(rows[0][0], dtype=np.double)

        if (intrinsics[1] > intrinsics[2]):
          num_horizontal += 1
          if num_horizontal == 1:
            first_horizontal_camera_id = images[filename][1]

    for filename in filename_list:
        assert(filename in images.keys())

        camera_id = images[filename][1]
        cursor.execute("SELECT params FROM cameras WHERE camera_id=?;", (camera_id,))
        rows = cursor.fetchall()
        assert(len(rows) == 1)
        intrinsics = np.fromstring(rows[0][0], dtype=np.double)

        if (intrinsics[1] < intrinsics[2]):
          num_vertical += 1
          if num_vertical == 1:
            first_vertical_camera_id = images[filename][1]

    print 'Horizontal, vertical, total: ', num_horizontal, num_vertical, len(filename_list)
    assert(num_horizontal + num_vertical == len(filename_list))

    num_changed = 0

    for filename in filename_list:
        assert(filename in images.keys())
        image_id = images[filename][0]

        camera_id = images[filename][1]
        cursor.execute("SELECT params FROM cameras WHERE camera_id=?;", (camera_id,))
        rows = cursor.fetchall()
        assert(len(rows) == 1)
        intrinsics = np.fromstring(rows[0][0], dtype=np.double)

        cursor.close()
        cursor = connection.cursor()

        # Update camera association of the image.
        if (intrinsics[1] > intrinsics[2]):
          if num_horizontal > 500:
            num_changed += 1
            new_params = [first_horizontal_camera_id, image_id]
            cursor.execute('UPDATE images SET camera_id = ? WHERE image_id = ?;', new_params)
        else:
          if num_vertical > 500:
            num_changed += 1
            new_params = [first_vertical_camera_id, image_id]
            cursor.execute('UPDATE images SET camera_id = ? WHERE image_id = ?;', new_params)

    print 'Updated', num_changed, 'database entries.'

    cursor.close()
    connection.commit()
    connection.close()

def main():
    db_file = 'database.db'
    images = get_images_from_db(db_file)

    output = defaultdict(list)
    for filename in os.listdir('images_upright/db'):
        if filename.endswith(".jpg"):
            img = PIL.Image.open('images_upright/db/' + filename)
            exif_data = img._getexif()
            output[exif_data[272] + '_f' + str(exif_data[37386][0])].append('db/' + filename)

    print 'Got', len(output), 'unique cameras and focal lengths.'

    for camera, files in output.iteritems():
        read_intrinsics_db(db_file, files, images)


if __name__ == "__main__":
    main()
