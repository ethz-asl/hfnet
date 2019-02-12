import sqlite3
import numpy as np
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm

from hfnet.datasets.colmap_utils.read_model import qvec2rotmat
from hfnet.utils.tools import Timer  # noqa: F401 (profiling)


colmap_cursors = {}

DummyImage = namedtuple(
    'DummyImage', ['shape'])
LocalDbItem = namedtuple(
    'LocalDbItem', ['landmark_ids', 'descriptors', 'keypoints'])
QueryInfo = namedtuple(
    'QueryInfo', ['name', 'model', 'width', 'height', 'K', 'dist'])
QueryItem = namedtuple(
    'QueryItem', ['global_desc', 'keypoints', 'local_desc'])


def get_cursor(name):
    global colmap_cursors
    if name not in colmap_cursors:
        colmap_cursors[name] = sqlite3.connect(name).cursor()
    return colmap_cursors[name]


def descriptors_from_colmap_db(cursor, image_id):
    cursor.execute(
            f'SELECT cols, data FROM descriptors WHERE image_id="{image_id}";')
    feature_dim, blob = next(cursor)
    desc = np.frombuffer(blob, dtype=np.uint8).reshape(-1, feature_dim)
    return desc


def keypoints_from_colmap_db(cursor, image_id):
    cursor.execute(
        f'SELECT cols, data FROM keypoints WHERE image_id="{image_id}";')
    cols, blob = next(cursor)
    kpts = np.frombuffer(blob, dtype=np.float32).reshape(-1, cols)
    return kpts


def dummy_iter(ids, images, cameras):
    """ Standard loaders (shared across the evaluation pipelines) require at
        least a dictionary with an item name and an image whose shape can be
        needed. To avoid reading images from disk unnecessarily, we create
        dummy images that have a shape but no data.
    """
    for i in ids:
        im = images[i]
        cam = cameras[im.camera_id]
        name = im.name
        yield {'name': Path(Path(name).parent, Path(name).stem).as_posix(),
               'image': DummyImage((cam.height, cam.width, 1))}


def build_localization_dbs(db_ids, images, cameras,
                           config_global=None, config_local=None):
    global_descriptors = None
    local_db = []

    db_iter = dummy_iter(db_ids, images, cameras)
    for i, (image_id, data) in tqdm(enumerate(zip(db_ids, db_iter))):
        # Global
        if config_global is not None:
            pred = config_global['predictor'](
                data['image'], data['name'], **config_global)
            desc = pred['global_descriptor']
            if global_descriptors is None:
                global_descriptors = np.empty((len(db_ids), desc.shape[0]))
            global_descriptors[i] = desc

        # Local
        if config_local is not None:
            db_item = images[image_id]
            valid = db_item.point3D_ids > 0
            kpts = db_item.xys[valid] - 0.5  # Colmap -> CV convention

            if 'predictor' in config_local:
                # Kind of hacky but that's for the sake of reusability
                config = config_local.copy()
                config['num_features'] = 0  # keep all features
                config['keypoint_predictor'] = lambda im, n, **kwargs: {
                    'keypoints': kpts, 'scores': None}
                pred = config_local['predictor'](
                    data['image'], data['name'], **config)
                desc = pred['descriptors']
            elif 'colmap_db' in config_local:
                cursor = get_cursor(config_local['colmap_db'])
                if config_local.get('broken_db', False):
                    db_image_id, = next(cursor.execute(
                        'SELECT image_id FROM images '
                        f'WHERE name="{db_item.name}";'))
                else:
                    db_image_id = image_id
                desc = descriptors_from_colmap_db(cursor, db_image_id)
                assert desc.shape[0] == len(valid)
                desc = desc[valid]
            else:
                raise ValueError('Local config does not contain predictor '
                                 f'or colmap db: {config_local}')

            local_db.append(
                LocalDbItem(db_item.point3D_ids[valid], desc, kpts))

    local_db = dict(zip(db_ids, local_db))
    return global_descriptors, local_db


def read_query_list(path, prefix=''):
    queries = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            (name, model, w, h), params = data[:4], data[4:]
            if model == 'SIMPLE_RADIAL':
                f, px, py, dist = params
                fx = fy = f
            elif model == 'PINHOLE':
                fx, fy, px, py = params
                dist = 0.0
            else:
                raise(ValueError, f'Unknown camera model: {model}')
            K = np.array([[float(fx), 0, float(px)-0.5],
                          [0, float(fy), float(py)-0.5],
                          [0, 0, 1]])
            name = str(Path(prefix, name))
            query = QueryInfo(name, model, int(w), int(h), K, float(dist))
            queries.append(query)
    return queries


def extract_query(data, info, config_global, config_local):
    # Global
    global_desc = config_global['predictor'](
            data['image'], data['name'], **config_global)['global_descriptor']

    # Local
    if 'predictor' in config_local:
        pred_local = config_local['predictor'](
            data['image'], data['name'], **config_local)
        kpts, local_desc = pred_local['keypoints'], pred_local['descriptors']
        scaling = (np.array([info.width, info.height])
                   / np.array(data['image'].shape[:2][::-1]))
        kpts = kpts * scaling
    elif 'colmap_db' in config_local:
        db_name = config_local.get(
            'colmap_db_queries', config_local['colmap_db'])
        cursor = get_cursor(db_name)
        db_query_name = info.name
        if config_local.get('broken_db', False):
            db_query_name = db_query_name.replace('jpg', 'png')
        if config_local.get('broken_paths', False):
            db_query_name = 'images/' + db_query_name
        query_id, = next(cursor.execute(
            f'SELECT image_id FROM images WHERE name="{db_query_name}";'))
        kpts = keypoints_from_colmap_db(cursor, query_id)[:, :2]
        local_desc = descriptors_from_colmap_db(cursor, query_id)
    else:
        raise ValueError('Local config does not contain predictor '
                         f'or colmap db: {config_local}')

    return QueryItem(global_desc, kpts, local_desc)


def colmap_image_to_pose(image):
    im_T_w = np.eye(4)
    im_T_w[:3, :3] = qvec2rotmat(image.qvec)
    im_T_w[:3, 3] = image.tvec
    w_T_im = np.linalg.inv(im_T_w)
    return w_T_im
