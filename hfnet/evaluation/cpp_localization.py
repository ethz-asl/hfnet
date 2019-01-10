import numpy as np
import cv2
import logging

from .utils.localization import LocResult


class CppLocalization:
    def __init__(self, db_ids, local_db, global_descriptors, images, points):
        import _hloc_cpp
        self.hloc = _hloc_cpp.HLoc()

        id_to_idx = {}
        old_to_new_kpt = {}
        for idx, i in enumerate(db_ids):
            keypoints = local_db[i].keypoints.T.astype(np.float32).copy()
            local_desc = local_db[i].descriptors.T.astype(np.float32).copy()
            global_desc = global_descriptors[idx].astype(np.float32).copy()
            # keypoints are NOT undistorted or nomalized
            idx = self.hloc.addImage(global_desc, keypoints, local_desc)

            id_to_idx[i] = idx
            old_to_new_kpt[i] = {
                k: j for j, k
                in enumerate(np.where(images[i].point3D_ids >= 0)[0])}

        for i, pt in points.items():
            observations = np.array(
                [[id_to_idx[im_id], old_to_new_kpt[im_id][kpt_id]]
                 for im_id, kpt_id in zip(pt.image_ids, pt.point2D_idxs)],
                dtype=np.int32)
            self.hloc.add3dPoint(
                pt.xyz.astype(np.float32).copy(), observations.copy())
        self.hloc.buildIndex()

    def localize(self, query_info, query_item, global_transf, local_transf):
        global_desc = global_transf(query_item.global_desc[np.newaxis])[0]
        local_desc = local_transf(query_item.local_desc)
        keypoints = cv2.undistortPoints(
            query_item.keypoints[np.newaxis], query_info.K,
            np.array([query_info.dist, 0, 0, 0]))[0]

        logging.info('Localizing image %s', query_info.name)
        ret = self.cpp_backend.localize(
            global_desc.astype(np.float32),
            keypoints.astype(np.float32).T.copy(),
            local_desc.astype(np.float32).T.copy())
        (success, num_components_total, num_components_tested,
         last_component_size, num_db_landmarks, num_matches,
         num_inliers, num_iters, global_ms, covis_ms, local_ms, pnp_ms) = ret

        result = LocResult(success, num_inliers, 0, np.eye(4))
        stats = {
            'success': success,
            'num_components_total': num_components_total,
            'num_components_tested': num_components_tested,
            'last_component_size': last_component_size,
            'num_db_landmarks': num_db_landmarks,
            'num_matches': num_matches,
            'num_inliers': num_inliers,
            'num_ransac_iters': num_iters,
            'timings': {
                'global': global_ms,
                'covis': covis_ms,
                'local': local_ms,
                'pnp': pnp_ms,
            }
        }
        return (result, stats)
