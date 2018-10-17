import numpy as np
import os
import sqlite3

def db_image_name_dict():
    # CHANGE DATABASE PATH HERE!
    connection = sqlite3.connect('test2.db')
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

    print max(image_ids)

    return name_to_image_id

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 0.00001:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])



def export_cameras():
    connection = sqlite3.connect('test2.db')
    cursor = connection.cursor()

    outfile = open("cameras.txt", "w")

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
        camera_id = row[0]
        assert(row[1] == 2)
        model = 'SIMPLE_RADIAL'
        width = row[2]
        height = row[3]
        params = np.fromstring(row[4], dtype=np.double)
        cameras[camera_id] = params
        print camera_id, model, params
        outfile.write('%d %s %f %f ' %(camera_id, model, width, height))
        for x in params:
            outfile.write(str(x) + ' ')
        outfile.write('\n')

    outfile.close()


def process(data, name_to_image_id,outfile):
    # colmap format
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    #print data[2], data[3], data[4], data[5]
    #print data[6], data[7], data[8]
    image_id = name_to_image_id[os.path.basename(data[0])]
    assert (image_id > 0)
    print data[0], image_id

    test = str(data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + data[5])
    q = np.fromstring(test, dtype=float, sep=' ')
    v = np.array([float(data[6]), float(data[7]), float(data[8])]).transpose()

    R = quat2mat(q);

    print R, R.transpose()

    tvec = v #R.dot(-v)

    print tvec

    outfile.write(str(image_id))
    outfile.write(' %s %s %s %s %f %f %f ' %(data[2],data[3],data[4],data[5],tvec[0],tvec[1],tvec[2]))
    outfile.write(str(image_id) + ' ')
    outfile.write(os.path.basename(data[0]) + '\n\n')

    # outfile.write(str(0))
    # outfile.write(' %s %s %s %s %s %s %s ' %(data[2],data[3],data[4],data[5],data[6],data[7],data[8]))
    # outfile.write(str(0) + ' ')
    # outfile.write(os.path.basename(data[0]) + '\n\n')
    # exit(0)


def main():
  print 'Reading DB'
  name_to_image_id = db_image_name_dict()

  print 'Exporting cameras'
  export_cameras()

  print 'Reading NVM'
  with open('aachen_cvpr2018_db.nvm') as f:
    line_num = 0
    total_num_images = 0
    num_images = 0

    outfile = open("images.txt", "w")

    for line in f:
      if line_num == 0 or line_num == 1:
          print 'Header, skip'
      elif line_num == 2:
           total_num_images = int(line)
      else:
          data = line.split(' ')
          process(data, name_to_image_id, outfile)
          num_images += 1
          if (num_images == total_num_images):
            return
      line_num += 1

    outfile.close()

if __name__ == "__main__":
    main()
