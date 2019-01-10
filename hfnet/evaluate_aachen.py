import logging
from pathlib import Path
import argparse
from pprint import pformat
import yaml
import numpy as np
from pyquaternion import Quaternion

from hfnet.evaluation.localization import Localization, evaluate
from hfnet.evaluation.loaders import export_loader
from hfnet.settings import EXPER_PATH


configs_global = {
    'netvlad': {
        'db_name': 'globaldb_netvlad.pkl',
        'experiment': 'netvlad/aachen',
        'predictor': export_loader,
        'has_keypoints': False,
        'has_descriptors': False,
        'pca_dim': 1024,
        'num_prior': 10,
    },
    'mnv': {
        'db_name': 'globaldb_mnv0.35.pkl',
        'experiment': 'mobilenetvlad_depth-0.35/aachen_resize-960_layer-7',
        'predictor': export_loader,
        'has_keypoints': False,
        'has_descriptors': False,
        'pca_dim': 1024,
        'num_prior': 10,
    },
    'hfnet': {
        'db_name': 'globaldb_hf_glm-bdd_lrsteps.pkl',
        'experiment':  'hfnet-shared_weights-unc_aug-photo_glm-bdd_lrsteps/aachen_resize-960',
        'predictor': export_loader,
        'has_keypoints': False,
        'has_descriptors': False,
        'pca_dim': 1024,
        'num_prior': 10,
    },
}

configs_local = {
    'superpoint': {
        'db_name': 'localdb_sp-nms4_fix-interp.pkl',
        'experiment': 'super_point_pytorch/aachen_resize-960',
        'predictor': export_loader,
        'has_keypoints': True,
        'has_descriptors': True,
        'binarize': False,
        'do_nms': True,
        'nms_thresh': 4,
        'num_features': 2000,
        'ratio_thresh': 0.9,
        'nms_refinement': False,
    },
    'hfnet': {
        'db_name': 'localdb_hf_glm-bdd_lrsteps.pkl',
        'experiment':  'hfnet-shared_weights-unc_aug-photo_glm-bdd_lrsteps/aachen_resize-960',
        'predictor': export_loader,
        'has_keypoints': True,
        'has_descriptors': True,
        'do_nms': True,
        'nms_thresh': 4,
        'num_features': 2000,
        'ratio_thresh': 0.9,
    },
    'sift': {
        'db_name': 'localdb_sift_raw.pkl',
        'colmap_db': 'aachen.db',
        'root': False,
        'ratio_thresh': 0.7,
    },
    'doap': {
        'db_name': 'localdb_doap_kpts-sp-nms4.pkl',
        'experiment': 'doap/aachen_resize-960',
        'predictor': export_loader,
        'keypoint_predictor': export_loader,
        'keypoint_config': {
            'experiment': 'super_point_pytorch/aachen_resize-960',
            'do_nms': True,
            'nms_thresh': 4,
        },
        'num_features': 2000,
        'ratio_thresh': 0.9,
    },
    'netvlad': {
        'db_name': 'localdb_netvlad_kpts-sp-nms4.pkl',
        'experiment': 'netvlad/aachen',
        'predictor': export_loader,
        'keypoint_predictor': export_loader,
        'keypoint_config': {
            'experiment': 'super_point_pytorch/aachen_resize-960',
            'do_nms': True,
            'nms_thresh': 4,
        },
        'num_features': 1000,
        'ratio_thresh': 0.7,
    }
}

config_pose = {
    'reproj_error': 10,
    'min_inliers': 12,
}

config_aachen = {
    'resize_max': 960,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('eval_name', type=str)
    parser.add_argument('--local_method', type=str)
    parser.add_argument('--global_method', type=str)
    parser.add_argument('--build_db', action='store_true')
    parser.add_argument('--queries', type=str, default='day_time')
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--export_poses', action='store_true')
    parser.add_argument('--cpp_backend', action='store_true')
    args = parser.parse_args()

    config = {
        'global': configs_global[args.global_method],
        'local': configs_local[args.local_method],
        'aachen': config_aachen,
        'pose': config_pose,
        'model': args.model,
        'max_iter': args.max_iter,
        'queries': args.queries,
        'use_cpp': args.cpp_backend,
    }
    logging.info(f'Configuration: \n'+pformat(config))

    logging.info('Evaluating Aachen with configuration: ')
    loc = Localization('aachen', args.model, config, build_db=args.build_db)

    query_file = f'{args.queries}_queries_with_intrinsics.txt'
    queries, query_dataset = loc.init_queries(query_file, config_aachen)

    logging.info('Starting evaluation')
    metrics, results = evaluate(
        loc, queries, query_dataset, max_iter=args.max_iter)
    logging.info('Evaluation metrics: \n'+pformat(metrics))

    output = {'config': config, 'metrics': metrics}
    output_dir = Path(EXPER_PATH, 'eval/aachen')
    output_dir.mkdir(exist_ok=True, parents=True)
    eval_path = Path(output_dir, f'{args.eval_name}.yaml')
    with open(eval_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False)

    if args.export_poses:
        poses_path = Path(output_dir, f'{args.eval_name}_poses.txt')
        with open(poses_path, 'w') as f:
            for query, result in zip(queries, results):
                query_T_w = np.linalg.inv(result.T)
                qvec_nvm = list(Quaternion(matrix=query_T_w))
                pos_nvm = query_T_w[:3, 3].tolist()
                line = f'{Path(query.name).name}'
                line += ' ' + ' '.join(map(str, qvec_nvm+pos_nvm))
                f.write(line+'\n')
