import argparse
import numpy as np
import os
import sqlite3

from internal import nvm_to_colmap_helper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", required=True)
    parser.add_argument("--nvm_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    return args


def db_image_name_dict(db_file):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    # Get a mapping between image ids and image names
    name_to_image_id = dict()
    image_ids = []
    cursor.execute('SELECT image_id, name FROM images;')
    for row in cursor:
        image_id = row[0]
        image_ids.append(image_id)
        name = row[1]
        name_to_image_id[name] = image_id

    return name_to_image_id


def export_cameras(db_file, output_dir):
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()

    outfile = open(os.path.join(output_dir, "cameras.txt"), "w")

    # In db file:
    # model INTEGER NOT NULL,
    # width INTEGER NOT NULL,
    # height INTEGER NOT NULL,
    # params BLOB,
    # prior_focal_length INTEGER NOT NULL)"""

    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]

    cameras = {}
    cursor.execute("SELECT camera_id, model, width, height, params FROM cameras;")
    for row in cursor:
        if row[1] != 2:
            # Wrong model, skip this camera.
            continue

        camera_id = row[0]
        assert(row[1] == 2)
        model = 'SIMPLE_RADIAL'
        width = row[2]
        height = row[3]
        params = np.fromstring(row[4], dtype=np.double)
        cameras[camera_id] = params
        outfile.write('%d %s %f %f ' %(camera_id, model, width, height))
        for x in params:
            outfile.write(str(x) + ' ')
        outfile.write('\n')

    outfile.close()


def process(nvm_data, name_to_image_id,outfile):
    # colmap format
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    # nvm format
    # q -> data[2], data[3], data[4], data[5]
    # p -> data[6], data[7], data[8]

    assert(nvm_data[0] in name_to_image_id), nvm_data[0]
    image_id = name_to_image_id[nvm_data[0]]

    assert (image_id > 0)

    q_nvm_str = str(nvm_data[2] + ' ' + nvm_data[3] + ' ' + nvm_data[4] + ' ' + nvm_data[5])
    q_nvm = np.fromstring(q_nvm_str, dtype=float, sep=' ')
    p_nvm = np.array([float(nvm_data[6]), float(nvm_data[7]), float(nvm_data[8])])

    p_colmap = nvm_to_colmap_helper.convert_nvm_pose_to_colmap_p(q_nvm, p_nvm)

    outfile.write(str(image_id))
    outfile.write(' %s %s %s %s %f %f %f ' %(nvm_data[2],nvm_data[3], \
        nvm_data[4],nvm_data[5],p_colmap[0],p_colmap[1],p_colmap[2]))
    outfile.write(str(image_id) + ' ')
    outfile.write(nvm_data[0] + '\n\n')


def main():
  args = parse_args()

  print 'Reading DB'
  name_to_image_id = db_image_name_dict(args.database_file)

  print 'Exporting cameras'
  export_cameras(args.database_file, args.output_dir)

  print 'Creating empty Points3D file'
  open(os.path.join(args.output_dir, 'points3D.txt'), 'w+').close()

  print 'Reading NVM'
  with open(args.nvm_file) as f:
    line_num = 0
    total_num_images = 0
    num_images = 0

    outfile = open(os.path.join(args.output_dir, "images.txt"), "w")

    for line in f:
      if line_num == 0 or line_num == 1:
          pass
      elif line_num == 2:
           total_num_images = int(line)
      else:
          data = line.split(' ')
          process(data, name_to_image_id, outfile)
          num_images += 1
          if (num_images == total_num_images):
            break
      line_num += 1

    outfile.close()

  print 'Done'


if __name__ == "__main__":
    main()
