import argparse
import numpy as np
import sqlite3
import os

from internal.db_handling import array_to_blob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", required=True)
    parser.add_argument("--nvm_file", required=True)
    args = parser.parse_args()
    return args

def db_image_name_dict(db_file):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    # cursor.execute("SELECT camera_id, model, width, height, params FROM cameras;")
    # for row in cursor:
    #     np_array = np.fromstring(row[4], dtype=np.double)
    #     print np_array
    #     print row[4],  np_array.tostring()
    #     print row[4].decode("utf-8").size()
    #     print np_array.tostring().size()
    # cursor.close()
    # exit(0)

    # Get a mapping between image ids and image names and camera ids.
    image_name_to_id_and_camera_id = dict()
    image_ids = []
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        camera_id = row[0]
        name = row[2]
        image_name_to_id_and_camera_id[name] = (image_id, camera_id)

    cursor.close()
    connection.close()

    return image_name_to_id_and_camera_id

def db_update_intrinsics(db_connection, camera_id, focal_length, radial_dist):
    cursor = db_connection.cursor()
    cursor.execute("SELECT params FROM cameras WHERE camera_id=?;", (camera_id,))
    rows = cursor.fetchall()
    assert(len(rows) == 1)
    intrinsics = np.fromstring(rows[0][0], dtype=np.double)
    cursor.close()

    # Update intrinsic values.
    intrinsics[0] = focal_length
    # See https://github.com/colmap/colmap/blob/master/src/base/reconstruction.cc#L839
    intrinsics[3] = -1 * radial_dist

    # Write intrinsics back.
    cursor = db_connection.cursor()

    # Python 3 version.
    #new_params = [intrinsics.tostring(), camera_id]

    new_params = [array_to_blob(intrinsics), camera_id]
    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;', new_params)
    cursor.close()


def process(nvm_data, image_name_to_id_and_camera_id, db_connection):
    image_filename = os.path.splitext(nvm_data[0])[0] + '.jpg'
    if image_filename.startswith("./"):
      image_filename = image_filename[2:]
    focal_length = float(nvm_data[1])
    radial_dist = float(nvm_data[9])

    assert(image_filename in image_name_to_id_and_camera_id), \
        ('No such image name in dict', image_filename)
    image_id, camera_id = image_name_to_id_and_camera_id[image_filename]

    print(image_filename, focal_length, radial_dist, \
        'maps to image_id', image_id, \
        'and camera_id', camera_id)

    db_update_intrinsics(db_connection, camera_id, focal_length, radial_dist)


def main():
  args = parse_args()

  print('Reading DB')
  image_name_to_id_and_camera_id = db_image_name_dict(args.database_file)

  connection = sqlite3.connect(args.database_file)

  print('Reading NVM')
  with open(args.nvm_file) as f:
    line_num = 0
    total_num_images = 0
    num_images = 0

    for line in f:
      if line_num == 0 or not line.strip():
          print('Skip line #' + str(line_num))
          line_num += 1
          continue
      else:
          if total_num_images == 0:
              total_num_images = int(line)
              line_num += 1
              continue
          data = line.split(' ')
          process(data, image_name_to_id_and_camera_id, connection)
          num_images += 1
          if (num_images == total_num_images):
            break
      line_num += 1

  print('Done parsing data for', num_images, 'images.')

  connection.commit()
  connection.close()


if __name__ == "__main__":
    main()
