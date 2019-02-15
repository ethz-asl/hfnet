import argparse
import numpy as np
import sqlite3

from internal.db_handling import array_to_blob, blob_to_array

# c0
# 868.99 0 525.94
# 0 866.06 420.04
# 0 0 1

# c1
# 873.38 0 529.32
# 0 876.49 397.27
# 0 0 1

cmu_intrinsics = {
    'c0': {
        'focal_length': (868.99 + 866.06) / 2,
        'cx': 525.94,
        'cy': 420.04,
    },
    'c1': {
        'focal_length': (873.38 + 876.49) / 2,
        'cx': 529.32,
        'cy': 397.27,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    args = parser.parse_args()
    return args


def update_db_intrinsics(db_connection, camera_id, focal_length, cx, cy):
    cursor = db_connection.cursor()
    cursor.execute('SELECT params FROM cameras WHERE camera_id=?;',
                   (camera_id,))
    data = cursor.fetchall()
    assert len(data) == 1
    intrinsics = blob_to_array(data[0][0], np.double)

    # Update intrinsic values.
    intrinsics[0] = focal_length
    intrinsics[1] = cx
    intrinsics[2] = cy

    cursor.execute('UPDATE cameras SET params = ? WHERE camera_id = ?;',
                   [array_to_blob(intrinsics), camera_id])
    cursor.close()


def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_file)
    cursor = connection.cursor()

    # Get a mapping between image ids and image names and camera ids.
    cursor.execute('SELECT image_id, camera_id, name FROM images;')
    for image_id, camera_id, name in cursor:
        if 'c0' in name:
            assert 'c1' not in name
            update_db_intrinsics(connection, camera_id, **cmu_intrinsics['c0'])
        else:
            assert 'c0' not in name
            update_db_intrinsics(connection, camera_id, **cmu_intrinsics['c1'])

    cursor.close()
    connection.commit()
    connection.close()


if __name__ == '__main__':
    main()
