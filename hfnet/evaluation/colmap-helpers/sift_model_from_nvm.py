import numpy as np
import os
from collections import defaultdict
import sqlite3
from tqdm import tqdm

from internal import nvm_to_colmap_helper

# Convert the txt model to bin before importing via GUI!:
# e.g. colmap model_converter --input_path from_nvm/ --output_path from_nvm/ --output_type BIN


# <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
# <Measurement> = <Image index> <Feature Index> <xy>

def image_id_from_name(image_name):
    connection = sqlite3.connect('aachen.db')
    cursor = connection.cursor()

    cursor.execute('SELECT image_id FROM images WHERE name=?;', (image_name,))
    data = cursor.fetchall()
    assert(len(data) == 1), image_name

    cursor.close()
    connection.close()

    return data[0][0]


def camera_params_from_image_id(image_id):
    connection = sqlite3.connect('aachen.db')
    cursor = connection.cursor()

    cursor.execute('SELECT camera_id FROM images WHERE image_id=?;', (image_id,))
    data = cursor.fetchall()
    assert(len(data) == 1)

    camera_id = data[0][0]
    cursor.execute("SELECT model, width, height, params FROM cameras WHERE camera_id=?;", (camera_id,))
    data = cursor.fetchall()
    assert(len(data) == 1)
    params = np.fromstring(data[0][3], dtype=np.double)

    cursor.close()
    connection.close()

    return camera_id, data[0][1], data[0][2], params


def read_keypoints_from_db(image_id):
    connection = sqlite3.connect('aachen.db')
    cursor = connection.cursor()

    cursor.execute('SELECT image_id, rows, cols, data FROM keypoints WHERE image_id=?;', (image_id,))
    data = cursor.fetchall()
    assert(len(data) == 1), image_id
    keypoints = np.fromstring(data[0][3], dtype=np.float32).reshape((-1, 4))

    cursor.close()
    connection.close()

    return keypoints


def export_image_data(image_data, image_id_to_keypoints):
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    out_images = open("images.txt", "w")
    out_cameras = open("cameras.txt", "w")

    image_id = 0

    for nvm_data in tqdm(image_data, total=len(image_data), unit="images"):
        q_nvm_str = str(nvm_data[2] + ' ' + nvm_data[3] + ' ' + nvm_data[4] + ' ' + nvm_data[5])
        q_nvm = np.fromstring(q_nvm_str, dtype=float, sep=' ')
        p_nvm = np.array([float(nvm_data[6]), float(nvm_data[7]), float(nvm_data[8])])

        p_colmap = nvm_to_colmap_helper.convert_nvm_pose_to_colmap_p(q_nvm, p_nvm)

        db_image_id = image_id_from_name(nvm_data[0])
        camera_id, width, height, params = camera_params_from_image_id(db_image_id)

        out_images.write(str(db_image_id))
        out_images.write(' %s %s %s %s %f %f %f ' %(nvm_data[2],nvm_data[3], \
            nvm_data[4],nvm_data[5],p_colmap[0],p_colmap[1],p_colmap[2]))
        out_images.write(str(camera_id) + ' ')
        out_images.write(nvm_data[0] + '\n')

        focal_length = float(nvm_data[1])
        distortion = -1 * float(nvm_data[9])

        # Sanity check of the params.
        #print params, focal_length, distortion

        # cameras.txt format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # e.g. 4479 SIMPLE_RADIAL 1067 1600 1094.05 533.5 800 -0.0819757
        # id, model, w, h, f, px, py, k
        out_cameras.write(str(camera_id) + ' SIMPLE_RADIAL %d %d %f %f %f %f\n' %(width, height, focal_length, params[1], params[2], distortion))

        # Now export the keypoints that correspond to 3D points.
        keypoints = read_keypoints_from_db(db_image_id)

        associated_keypoints = dict()
        if image_id in image_id_to_keypoints:
            matching_keypoints = image_id_to_keypoints[image_id];

            for matching_keypoint in matching_keypoints:
                point_index = matching_keypoint[3]
                kp_index = matching_keypoint[0]

                assert(kp_index not in associated_keypoints)
                associated_keypoints[kp_index] = point_index

                # Sanity check of the keypoint locations.
                #print matching_keypoint[1:3]
                #print keypoints[kp_index, 0:2]

        for kp_index, keypoint in enumerate(keypoints):
            if kp_index in associated_keypoints:
                point_idx = associated_keypoints[kp_index]
            else:
                point_idx = -1
            out_images.write('%f %f %d ' %(keypoint[0], keypoint[1], point_idx))

        out_images.write('\n')

        image_id += 1

    out_images.close()


def main():
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    out_points = open("points3D.txt", "w")

    image_idx_to_keypoints = defaultdict(list)
    image_idx_to_db_image_id = []
    image_data = []

    with open('aachen_cvpr2018_db.nvm') as f:
        line_num = 0
        total_num_images = 0
        total_num_points = 0

        num_images = 0
        num_points = 0

        for line in f:
          if line_num == 0 or line_num == 1:
              line_num += 1
              pass
          elif line_num == 2:
              total_num_images = int(line)
              break

        for i, line in enumerate(f):
            if i == total_num_images:
                break
            line_list = line.split(' ')
            image_data.append(line_list)
            image_idx_to_db_image_id.append(image_id_from_name(line_list[0]))


        for line in f:
            # Skip empty lines and read the one that actually contains the
            # number of 3D points in the NVM file.
            if line:
                total_num_points = int(line)
                break

        print 'Will export', total_num_points, '3D point entries.'
        for point_idx, line in tqdm(enumerate(f), total=total_num_points, unit="pts"):
            xyz = np.array(line.split()[0:3]).astype(np.float)
            rgb = np.array(line.split()[3:6]).astype(np.int)
            num_observations = int(line.split()[6])

            out_points.write('%d %.3f %.3f %.3f %d %d %d 1 ' %(point_idx, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]))

            for j in range(0, num_observations):
                # Offset + 4 values per observation.
                start_index = 7 + 4 * j
                observation = line.split()[start_index : start_index + 4]

                # In NVM, keypoints and images are indexed from 0.
                img_index = int(observation[0])
                kp_index = int(observation[1])
                image_idx_to_keypoints[img_index].append((kp_index, float(observation[2]), float(observation[3]), point_idx))

                db_image_id = image_idx_to_db_image_id[img_index]

                out_points.write('%d %d ' %(db_image_id, kp_index))

            out_points.write('\n')

    out_points.close()
    print 'Points3D done. Now exporting images.'
    export_image_data(image_data, image_idx_to_keypoints)

    print 'Done.'



if __name__ == "__main__":
    main()
