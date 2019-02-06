import argparse
import numpy as np
import sqlite3
import os

# c0
# 868.99 0 525.94
# 0 866.06 420.04
# 0 0 1

# c1
# 873.38 0 529.32
# 0 876.49 397.27
# 0 0 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", required=True)
    args = parser.parse_args()
    return args

def update_intrinsics_in_cmu_db(db_file):
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
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        camera_id = row[0]
        name = row[2]

        if "c0" in name:
          assert("c1" not in name)
          print "c0", name
          db_update_intrinsics_of_camera_id(connection, camera_id, (868.99 + 866.06) / 2.0, 525.94, 420.04)
        else:
          assert("c0" not in name)
          assert("c1" in name)
          print "c1", name
          db_update_intrinsics_of_camera_id(connection, camera_id, (873.38 + 876.49) / 2.0, 529.32, 397.27)

    cursor.close()

    connection.commit()
    connection.close()


def db_update_intrinsics_of_camera_id(db_connection, camera_id, focal_length, cx, cy):
    cursor = db_connection.cursor()
    cursor.execute("SELECT params FROM cameras WHERE camera_id=?;", (camera_id,))
    rows = cursor.fetchall()
    assert(len(rows) == 1)
    intrinsics = np.fromstring(rows[0][0], dtype=np.double)
    cursor.close()

    # Update intrinsic values.
    intrinsics[0] = focal_length
    intrinsics[1] = cx
    intrinsics[2] = cy
    # See https://github.com/colmap/colmap/blob/master/src/base/reconstruction.cc#L839

    # Write intrinsics back.
    cursor = db_connection.cursor()

    # Python 3 version.
    #new_params = [intrinsics.tostring(), camera_id]

    new_params = [np.getbuffer(intrinsics), camera_id]
    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;', new_params)
    cursor.close()


def main():
  args = parse_args()

  print 'Reading DB'
  update_intrinsics_in_cmu_db(args.database_file)


if __name__ == "__main__":
    main()
