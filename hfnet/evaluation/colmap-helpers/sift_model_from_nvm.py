import numpy as np
import os
from collections import defaultdict

from internal import nvm_to_colmap_helper

# <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
# <Measurement> = <Image index> <Feature Index> <xy>

def export_image_data(image_data, image_id_to_keypoints):
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    out_images = open("images.txt", "w")

    image_id = 0

    for nvm_data in image_data:
        print nvm_data
        
        q_nvm_str = str(nvm_data[2] + ' ' + nvm_data[3] + ' ' + nvm_data[4] + ' ' + nvm_data[5])
        q_nvm = np.fromstring(q_nvm_str, dtype=float, sep=' ')
        p_nvm = np.array([float(nvm_data[6]), float(nvm_data[7]), float(nvm_data[8])])

        p_colmap = nvm_to_colmap_helper.convert_nvm_pose_to_colmap_p(q_nvm, p_nvm)

        out_images.write(str(image_id))
        out_images.write(' %s %s %s %s %f %f %f ' %(nvm_data[2],nvm_data[3], \
            nvm_data[4],nvm_data[5],p_colmap[0],p_colmap[1],p_colmap[2]))
        out_images.write(str(image_id) + ' ')
        out_images.write(nvm_data[0] + '\n')

        image_id += 1


def main():
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    out_points = open("points3D.txt", "w")

    image_id_to_keypoints = defaultdict(list)
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
            image_data.append(line.split(' '))
            if i == total_num_images:
                break

        for line in f:
            # Skip empty lines and read the one that actually contains the num.
            if line:
                total_num_points = int(line)
                break

        print 'Will read', total_num_points, 'lines.'
        for i, line in enumerate(f):
            xyz = np.array(line.split()[0:3]).astype(np.float)
            rgb = np.array(line.split()[3:6]).astype(np.int)
            num_observations = int(line.split()[6])

            out_points.write('%d %f %f %f %d %d %d 0 ' %(i, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]))

            for j in range(0, num_observations):
                # Offset + 4 values per observation.
                start_index = 7 + 4 * j
                observation = line.split()[start_index : start_index + 4]

                kp_index = len(image_id_to_keypoints[int(observation[0])])
                image_id_to_keypoints[int(observation[0])].append((float(observation[2]), float(observation[3])))

                out_points.write('%d %d ' %(int(observation[0]), kp_index))

            out_points.write('\n')

            if i == 100:
                break

    out_points.close()
    print 'Points3D done. Now exporting images.'
    export_image_data(image_data, image_id_to_keypoints)

    print 'Done.'



if __name__ == "__main__":
    main()
