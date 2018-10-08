# -*- coding: utf-8 -*-

import math
import numpy as np
import os
import random
import sys
import glob
import cv2
import tensorflow as tf

from .spatial_transformer import inplane_inverse_warp


def inverse_warp_view_2_to_1(heatmaps2, depths2, depths1, c2Tc1s, K1, K2, inv_thetas1, thetas2, depth_thresh=0.5, get_warped_depth=False):
    # compute warping xy coordinate from view1 to view2
    # Args
    #   depths1: [B,H,W,1] tf.float32
    #   c2Tc1s: [B,4,4] tf.float32
    #   K1,K2: [B,3,3] tf.float32
    #   inv_thetas1: [B,3,3] tf.float32
    #   thetas2: [B,3,3] tf.float32
    # Return
    #   heatmaps1w : [B,H,W,1] tf.float32 warped heatmaps from camera2 to camera1
    #   visible_masks1 : [B,H,W,1] tf.float32 visible masks on camera1
    #   xy_u2 : [B,H,W,2] tf.float32 xy-coords-maps from camera1 to camera2

    def norm_meshgrid(batch, height, width, is_homogeneous=True):
        """Construct a 2D meshgrid.

        Args:
            batch: batch size
            height: height of the grid
            width: width of the grid
            is_homogeneous: whether to return in homogeneous coordinates
        Returns:
            x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
        """
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                      tf.transpose(tf.expand_dims(
                          tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                      tf.ones(shape=tf.stack([1, width])))
        if is_homogeneous:
            ones = tf.ones_like(x_t)
            coords = tf.stack([x_t, y_t, ones], axis=0)
        else:
            coords = tf.stack([x_t, y_t], axis=0)
        coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
        return coords

    def norm_xy_coords(xyz, height, width):
        # xyz: [B,>=3,N], tf.float32 xyz[:,0] = x, xyz[:,1]=y
        # suppose 0<=x<width, 0<=y<height
        # outputs range will be [-1,1]
        x_t = tf.slice(xyz, [0,0,0], [-1,1,-1])
        y_t = tf.slice(xyz, [0,1,0], [-1,1,-1])
        z_t = tf.slice(xyz, [0,2,0], [-1,-1,-1])
        x_t = 2 * (x_t / tf.cast(width-1,tf.float32)) - 1.0
        y_t = 2 * (y_t / tf.cast(height-1, tf.float32)) - 1.0
        n_xyz = tf.concat([x_t, y_t, z_t], axis=1)
        return n_xyz

    def unnorm_xy_coords(xyz, height, width):
        # xyz: [B,>=3,N], tf.float32 xyz[:,0] = x, xyz[:,1]=y
        # suppose -1<=x<=1, -1<=y<=1
        # outputs range will be [0,width) or [0,height)

        x_t = tf.slice(xyz, [0,0,0], [-1,1,-1])
        y_t = tf.slice(xyz, [0,1,0], [-1,1,-1])
        z_t = tf.slice(xyz, [0,2,0], [-1,-1,-1])

        x_t = (x_t+1.0) * 0.5 * tf.cast(width-1, tf.float32)
        y_t = (y_t+1.0) * 0.5 * tf.cast(height-1, tf.float32)

        u_xyz = tf.concat([x_t, y_t, z_t], axis=1)
        return u_xyz

    with tf.name_scope('WarpCoordinates'):
        if K2 is None:
            K2 = K1 # use same intrinsic matrix

        eps = 1e-6
        batch_size = tf.shape(depths1)[0]
        height1 = tf.shape(depths1)[1]
        width1 = tf.shape(depths1)[2]

        inv_K1 = tf.matrix_inverse(K1)
        right_col = tf.zeros([batch_size,3,1], dtype=tf.float32)
        bottom_row = tf.tile(tf.constant([0,0,0,1], dtype=tf.float32, shape=[1,1,4]), [batch_size,1,1])
        K2_4x4 = tf.concat([K2, right_col], axis=2)
        K2_4x4 = tf.concat([K2_4x4, bottom_row], axis=1)

        xy_n1 = norm_meshgrid(batch_size, height1, width1) # [B,3,H,W]
        xy_n1 = tf.reshape(xy_n1, [batch_size, 3, -1]) # [B,3,N], N=H*W

        # Inverse inplane transformation on camera1
        if inv_thetas1 is not None:
            xy_n1 = tf.matmul(inv_thetas1, xy_n1)
            z_n1 = tf.slice(xy_n1, [0,2,0],[-1,1,-1])
            xy_n1 = xy_n1 / (z_n1+eps)

        # SE(3) transformation : pixel1 to camera1
        Z1 = tf.reshape(depths1, [batch_size, 1, -1])
        xy_u1 = unnorm_xy_coords(xy_n1, height=height1, width=width1)
        XYZ1 = tf.matmul(inv_K1, xy_u1) * Z1
        ones = tf.ones([batch_size, 1, height1*width1])
        XYZ1 = tf.concat([XYZ1, ones], axis=1)

        # SE(3) transformation : camera1 to camera2 to pixel2
        proj_T = tf.matmul(K2_4x4, c2Tc1s)
        xyz2 = tf.matmul(proj_T, XYZ1)
        z2 = tf.slice(xyz2, [0,2,0], [-1,1,-1])
        reproj_depths = tf.reshape(z2, [batch_size, 1, height1, width1])
        reproj_depths = tf.transpose(reproj_depths, perm=[0,2,3,1]) # [B,H,W,1]

        xy_u2 = tf.slice(xyz2, [0,0,0],[-1,3,-1]) / (z2+eps)

        # Inplane transformation on camera2
        if thetas2 is not None:
            xy_n2 = norm_xy_coords(xy_u2, height1, width1)
            xy_n2 = tf.matmul(thetas2, xy_n2)
            z_n2 = tf.slice(xy_n2,[0,2,0], [-1,1,-1])
            xy_n2 = xy_n2 / (z_n2+eps)
            xy_u2 = unnorm_xy_coords(xy_n2, height1, width1)
        xy_u2 = tf.slice(xy_u2, [0,0,0],[-1,2,-1]) # discard third dim
        xy_u2 = tf.reshape(xy_u2, [batch_size, 2, height1, width1])
        xy_u2 = tf.transpose(xy_u2, perm=[0, 2, 3, 1]) # [B,H,W,2]

        heatmaps1w, depths1w = bilinear_sampling(heatmaps2, xy_u2, depths2) # it is not correct way to check depth consistency but it works well practically
        visible_masks = get_visibility_mask(xy_u2)
        camfront_masks = tf.cast(tf.greater(reproj_depths, tf.zeros((), reproj_depths.dtype)), tf.float32)
        nonocc_masks = tf.cast(tf.less(tf.squared_difference(depths1w, depths1), depth_thresh**2), tf.float32)
        visible_masks = visible_masks * camfront_masks * nonocc_masks # take logical_and
        heatmaps1w = heatmaps1w * visible_masks

        if get_warped_depth:
            return heatmaps1w, visible_masks, xy_u2, depths1w
        else:
            return heatmaps1w, visible_masks, xy_u2

def get_angle_colorbar():
    hue = np.arange(360)[:,None, None].astype(np.float32)
    ones = np.ones_like(hue)
    hsv = np.concatenate([hue, ones, ones], axis=-1)
    colorbar = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    colorbar = np.squeeze(colorbar)
    return colorbar

def get_degree_maps(ori_maps):
    # ori_maps : [B,H,W,2], consist of cos, sin response
    cos_maps = tf.slice(ori_maps, [0,0,0,0], [-1,-1,-1,1])
    sin_maps = tf.slice(ori_maps, [0,0,0,1], [-1,-1,-1,1])
    atan_maps = tf.atan2(sin_maps, cos_maps)
    angle2rgb = tf.constant(get_angle_colorbar())
    degree_maps = tf.cast(tf.clip_by_value(atan_maps*180/np.pi+180, 0, 360), tf.int32)
    degree_maps = tf.gather(angle2rgb, degree_maps[...,0])
    return degree_maps, atan_maps

# def inverse_warp_view_1_to_2(photos1, depths1, depths2, c1Tc2s, intrinsics_3x3, thetas1=None, thetas2=None, depth_thresh=0.5):
#     if thetas1 is not None:
#         # inverse in-plane transformation
#         photo_depths1 = tf.concat([photos1, depths1], axis=-1)
#         inwarp_photo_depths1, _ = inplane_inverse_warp(photo_depths1, thetas1)
#         photos1 = tf.slice(inwarp_photo_depths1, [0,0,0,0],[-1,-1,-1,1])
#         depths1 = tf.slice(inwarp_photo_depths1, [0,0,0,1],[-1,-1,-1,1])
#     # projective inverse transformation
#     photos2w, visible_masks2 = projective_inverse_warp(photos1, depths2, c1Tc2s,
#                                                     intrinsics_3x3, depths1, depth_thresh)
#     projective_visible_masks2 = tf.identity(visible_masks2)
#     if thetas2 is not None:
#         # inverse in-plane transformation
#         photo_masks2 = tf.concat([photos2w, visible_masks2], axis=-1)
#         inwarp_photo_masks2, _ = inplane_inverse_warp(photo_masks2, thetas2)
#         photos2w = tf.slice(inwarp_photo_masks2, [0,0,0,0], [-1,-1,-1,1])
#         visible_masks2 = tf.slice(inwarp_photo_masks2, [0,0,0,1], [-1,-1,-1,1])
#     visible_masks2 = tf.cast(tf.greater(visible_masks2, 0.5), tf.float32)
#     return photos2w, visible_masks2, projective_visible_masks2

def end_of_frame_masks(height, width, radius, dtype=tf.float32):
    eof_masks = tf.ones(tf.stack([1,height-2*radius,width-2*radius,1]), dtype=dtype)
    eof_masks = tf.pad(eof_masks, [[0,0],[radius,radius],[radius,radius],[0,0]])
    return eof_masks

def morphology_closing(inputs, dilate_ksize=3, erode_ksize=5):
    curr_in = inputs
    if dilate_ksize > 1:
        curr_in = tf.nn.max_pool(curr_in, [1,dilate_ksize,dilate_ksize,1],
                            strides=[1,1,1,1], padding='SAME')
    if erode_ksize > 1:
        curr_in = -tf.nn.max_pool(-curr_in, [1,erode_ksize,erode_ksize,1],
                            strides=[1,1,1,1], padding='SAME')
    return curr_in

def batch_gather_keypoints(inputs, batch_inds, kpts, xy_order=True):
    # kpts: [N,2] x,y or y,x
    # batch_inds: [N]
    # outputs = inputs[b,y,x]
    if xy_order:
        with tf.device('/cpu:0'):
            kp_x, kp_y = tf.split(kpts, 2, axis=1)
    else:
        with tf.device('/cpu:0'):
            kp_y, kp_x = tf.split(kpts, 2, axis=1)
    if len(batch_inds.get_shape().as_list()) == 1:
        batch_inds = batch_inds[:,None]
    byx = tf.concat([batch_inds, kp_y, kp_x], axis=1)
    outputs = tf.gather_nd(inputs, byx)
    return outputs

# def coordinate_se3_warp(kpts1, batch_inds, intrinsics_3x3, c2Tc1s, depths1, visible_masks2):
#     # kpts1: [N,2] int32 (x,y)
#     # batch_inds: [N,] int32 [0,batch_size)
#     # intrinsics_3x3: [3,3] float32
#     # c2Tc1s: [B,4,4] float32
#     # depths1: [B,H,W,1] float32
#     with tf.name_scope('XY_SE3_WARP'):
#         N = tf.shape(kpts1)[0]
#         # gather Z
#         kp_x, kp_y = tf.split(kpts1, 2, axis=1)
#         byx = tf.concat([batch_inds[:,None], kp_y, kp_x], axis=1) # [N,3]
#         kp_z = tf.gather_nd(depths1, byx) # [N,1]

#         # pix2cam
#         pix_kpts = tf.transpose(tf.cast(kpts1, tf.float32)) # [2,N]
#         ones = tf.ones([1,N], dtype=tf.float32)
#         hpix_kpts = tf.concat([pix_kpts, ones], axis=0) # [3,N]
#         inv_intrinsics = tf.matrix_inverse(intrinsics_3x3) # [3,3]
#         cam_kpts = tf.matmul(inv_intrinsics, hpix_kpts) * tf.transpose(kp_z) # [3,3]*[3,N]*[1,N] = [3,N]
#         hcam_kpts = tf.concat([cam_kpts, ones], axis=0) # [4,N]

#         # projection matrix
#         gathered_c2Tc1s = tf.gather(c2Tc1s, batch_inds) # [B,4,4],[N] --> [N,4,4]
#         intrinsics_4x4 = tf.concat([intrinsics_3x3, tf.zeros([3,1])], axis=1)
#         intrinsics_4x4 = tf.concat([intrinsics_4x4, tf.constant([0.,0.,0.,1.], shape=[1,4])], axis=0)
#         intrinsics_4x4 = tf.tile(intrinsics_4x4[None], [N,1,1]) # [N,4,4]
#         projT = tf.matmul(intrinsics_4x4, gathered_c2Tc1s) # [N,4,4]

#         # cam2pix
#         hcam_kpts = tf.transpose(hcam_kpts)[...,None] # [4,N]->[N,4,1]
#         hcam_kpts_w = tf.matmul(projT, hcam_kpts)[:,:,0] # [N,4,4]*[N,4,1]-->[N,4]

#         x_u = tf.slice(hcam_kpts_w, [0,0],[-1,1])
#         y_u = tf.slice(hcam_kpts_w, [0,1],[-1,1])
#         z_u = tf.slice(hcam_kpts_w, [0,2],[-1,1])

#         kp_xw = x_u / (z_u+1e-6)
#         kp_yw = y_u / (z_u+1e-6)
#         # kpts_w = tf.concat([kp_xw, kp_yw], axis=1) # [N,2]

#         # check visibility
#         x0 = tf.cast(tf.floor(kp_xw), tf.int32)
#         x1 = x0 + 1
#         y0 = tf.cast(tf.floor(kp_yw), tf.int32)
#         y1 = y0 + 1

#         height = tf.shape(visible_masks2)[1]
#         width = tf.shape(visible_masks2)[2]
#         inside_x = tf.logical_and(tf.greater_equal(x0, 0), tf.less(x1, width))
#         inside_y = tf.logical_and(tf.greater_equal(y0, 0), tf.less(y1, height))
#         visibility = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32)

#         kp_xw_safe = tf.clip_by_value(tf.cast(tf.round(kp_xw), tf.int32), 0, width-1)
#         kp_yw_safe = tf.clip_by_value(tf.cast(tf.round(kp_yw), tf.int32), 0, height-1)
#         kpts_w_safe = tf.concat([kp_xw_safe, kp_yw_safe], axis=1) # [N,1] tf.int32

#         byx = tf.concat([batch_inds[:,None], kp_yw_safe, kp_xw_safe], axis=1) # [N,3]
#         gather_visbility = tf.gather_nd(visible_masks2, byx) # [N,1]
#         # print('## ', visible_masks2.shape, byx.shape, gather_visbility.shape)
#         visibility = visibility * gather_visbility # [N,1] tf.float32
#         visibility = tf.squeeze(visibility, 1)  # [N, ] tf.float32
#         return kpts_w_safe, visibility

def coordinate_se3_warp(kpts1, batch_inds, intrinsics_3x3, c2Tc1s, depths1, visible_masks2):
    # kpts1: [N,2] int32 (x,y)
    # batch_inds: [N,] int32 [0,batch_size)
    # intrinsics_3x3: [B,3,3] float32
    # c2Tc1s: [B,4,4] float32
    # depths1: [B,H,W,1] float32
    with tf.name_scope('XY_SE3_WARP'):
        N = tf.shape(kpts1)[0]
        # gather Z
        kp_x, kp_y = tf.split(kpts1, 2, axis=1)
        byx = tf.concat([batch_inds[:,None], kp_y, kp_x], axis=1) # [N,3]
        kp_z = tf.gather_nd(depths1, byx) # [N,1]

        # pix2cam
        pix_kpts = tf.transpose(tf.cast(kpts1, tf.float32)) # [2,N]
        ones = tf.ones([1,N], dtype=tf.float32)
        hpix_kpts = tf.concat([pix_kpts, ones], axis=0) # [3,N]
        hpix_kpts = tf.expand_dims(tf.transpose(hpix_kpts), axis=-1) # [N,3,1]
        inv_intrinsics = tf.matrix_inverse(intrinsics_3x3) # [B,3,3]
        gatherd_inv_intrinsics = tf.gather(inv_intrinsics, batch_inds) # [B,3,3],[N]-->[N,3,3]
        cam_kpts = tf.squeeze(tf.matmul(gatherd_inv_intrinsics, hpix_kpts), axis=-1)  # [N,3,3]x[N,3,1] --> [N,3,1] --> [N,3]
        cam_kpts = tf.transpose(cam_kpts*kp_z) # [3,N]
        hcam_kpts = tf.concat([cam_kpts, ones], axis=0) # [4,N]

        # projection matrix
        gathered_c2Tc1s = tf.gather(c2Tc1s, batch_inds) # [B,4,4],[N] --> [N,4,4]

        batch = tf.shape(intrinsics_3x3)[0]
        right_col = tf.zeros([batch,3,1], dtype=tf.float32)
        bottom_row = tf.tile(tf.constant([0,0,0,1], dtype=tf.float32, shape=[1,1,4]), [batch,1,1])
        intrinsics_4x4 = tf.concat([intrinsics_3x3, right_col], axis=2)
        intrinsics_4x4 = tf.concat([intrinsics_4x4, bottom_row], axis=1) # [B,4,4]
        gathered_intrinsics_4x4 = tf.gather(intrinsics_4x4, batch_inds) # [B,4,4],[N] --> [N,4,4]

        projT = tf.matmul(gathered_intrinsics_4x4, gathered_c2Tc1s) # [N,4,4]

        # cam2pix
        hcam_kpts = tf.transpose(hcam_kpts)[...,None] # [4,N]->[N,4,1]
        hcam_kpts_w = tf.matmul(projT, hcam_kpts)[:,:,0] # [N,4,4]*[N,4,1]-->[N,4]

        x_u = tf.slice(hcam_kpts_w, [0,0],[-1,1])
        y_u = tf.slice(hcam_kpts_w, [0,1],[-1,1])
        z_u = tf.slice(hcam_kpts_w, [0,2],[-1,1])

        kp_xw = x_u / (z_u+1e-6)
        kp_yw = y_u / (z_u+1e-6)
        # kpts_w = tf.concat([kp_xw, kp_yw], axis=1) # [N,2]

        # check visibility
        x0 = tf.cast(tf.floor(kp_xw), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(kp_yw), tf.int32)
        y1 = y0 + 1

        height = tf.shape(visible_masks2)[1]
        width = tf.shape(visible_masks2)[2]
        inside_x = tf.logical_and(tf.greater_equal(x0, 0), tf.less(x1, width))
        inside_y = tf.logical_and(tf.greater_equal(y0, 0), tf.less(y1, height))
        visibility = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32)

        kp_xw_safe = tf.clip_by_value(tf.cast(tf.round(kp_xw), tf.int32), 0, width-1)
        kp_yw_safe = tf.clip_by_value(tf.cast(tf.round(kp_yw), tf.int32), 0, height-1)
        kpts_w_safe = tf.concat([kp_xw_safe, kp_yw_safe], axis=1) # [N,1] tf.int32

        byx = tf.concat([batch_inds[:,None], kp_yw_safe, kp_xw_safe], axis=1) # [N,3]
        gather_visbility = tf.gather_nd(visible_masks2, byx) # [N,1]
        # print('## ', visible_masks2.shape, byx.shape, gather_visbility.shape)
        visibility = visibility * gather_visbility # [N,1] tf.float32
        visibility = tf.squeeze(visibility, 1)  # [N, ] tf.float32
        return kpts_w_safe, visibility
def find_hard_negative_from_myself(feats):
    # feats.shape = [B,K,D]
    K = tf.shape(feats)[1]

    feats1_mat = tf.expand_dims(feats, axis=2) # [B,K,D] --> [B,K,1,D]
    feats2_mat = tf.expand_dims(feats, axis=1) # [B,L,D] --> [B,1,L,D]
    feats1_mat = tf.tile(feats1_mat, [1,1,K,1]) # [B,K,L,D]
    feats2_mat = tf.tile(feats2_mat, [1,K,1,1]) # [B,K,L,D]

    distances = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [B,K,K]
    myself = tf.eye(K)[None] * 1e5
    distances = distances + myself
    min_dist = tf.reduce_min(distances, axis=2) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2 [B,K]
    arg_min = tf.argmin(distances, axis=2, output_type=tf.int32)

    return min_dist, arg_min, distances

# def batch_nearest_neighbors_less_memory(feats1, batch_inds1, num_kpts1, feats2, batch_inds2, num_kpts2, batch_size, num_parallel=1, back_prop=False):
#     # feats1 = [B*K1, D], feats2 = [B*K2,D]
#     # batch_inds = [B*K,] takes [0,batch_size)
#     # num_kpts: [B,] tf.int32
#     # outputs = min_dist, arg_min [B*K]

#     N1 = tf.shape(feats1)[0]
#     batch_offsets1 = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts1)], axis=0)
#     ta_dist1 = tf.TensorArray(dtype=tf.float32, size=N1)
#     ta_inds1 = tf.TensorArray(dtype=tf.int32, size=N1)

#     N2 = tf.shape(feats2)[0]
#     batch_offsets2 = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts2)], axis=0)
#     ta_dist2 = tf.TensorArray(dtype=tf.float32, size=N1)
#     ta_inds2 = tf.TensorArray(dtype=tf.int32, size=N1)

#     init_state = (0, ta_dist1, ta_inds1, ta_dist2, ta_inds2)
#     condition = lambda i, _, _2: i < batch_size

#     def body(i, ta_dist, ta_inds):
#         pass


def find_hard_negative_from_myself_less_memory(feats, batch_inds, num_kpts, batch_size, num_parallel=1, back_prop=False):
    # feats = [B*K, D]
    # batch_inds = [B*K,] takes [0,batch_size)
    # num_kpts: [B,] tf.int32
    # outputs = min_dist, arg_min [B*K]
    N = tf.shape(feats)[0]
    batch_offsets = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts)], axis=0)
    ta_dist = tf.TensorArray(dtype=tf.float32, size=N)
    ta_inds = tf.TensorArray(dtype=tf.int32, size=N)
    init_state = (0, ta_dist, ta_inds)
    condition = lambda i, _, _2: i < batch_size

    def body(i, ta_dist, ta_inds):
        curr_inds = tf.cast(tf.reshape(tf.where(tf.equal(batch_inds, i)), [-1]), tf.int32)
        is_empty = tf.equal(tf.size(curr_inds), 0)
        curr_inds = tf.cond(is_empty, lambda: tf.constant([0,], dtype=tf.int32), lambda: curr_inds) # if empty use dammy index but all results are ignored
        curr_feats = tf.gather(feats, curr_inds) # [K,D]
        K = tf.shape(curr_feats)[0]
        feats1_mat = tf.expand_dims(curr_feats, axis=1) # [K,1,D]
        feats2_mat = tf.expand_dims(curr_feats, axis=0) # [1,K,D]
        feats1_mat = tf.tile(feats1_mat, [1,K,1]) # [K,K,D]
        feats2_mat = tf.tile(feats2_mat, [K,1,1])
        distances = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [K,K]
        myself = tf.eye(K) * 1e5
        distances = distances + myself # avoid to pickup myself
        min_dist = tf.reduce_min(distances, axis=1) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2 [B,K]
        arg_min = tf.argmin(distances, axis=1, output_type=tf.int32)

        offset = batch_offsets[i]
        arg_min = arg_min + offset

        ta_dist = tf.cond(is_empty, lambda: ta_dist, lambda: ta_dist.scatter(curr_inds, min_dist))
        ta_inds = tf.cond(is_empty, lambda: ta_inds, lambda: ta_inds.scatter(curr_inds, arg_min))
        return i+1, ta_dist, ta_inds
#         return i+1, ta_dist.scatter(curr_inds, min_dist), ta_inds.scatter(curr_inds, arg_min)

    n, ta_dist_final, ta_inds_final = tf.while_loop(condition, body, init_state,
                                                   parallel_iterations=num_parallel,
                                                   back_prop=back_prop)
    min_dist = ta_dist_final.stack()
    min_inds = ta_inds_final.stack()
    return min_dist, min_inds

def find_random_hard_negative_from_myself_with_geom_constrain_less_memory(num_pickup, feats, feats_warp, kpts_warp, batch_inds, num_kpts, batch_size, geom_sq_thresh, num_parallel=1, back_prop=False):
    # find nearest ref-feats index
    # feats = [B*K, D] feature on image1
    # feats_warp = [B*K, D] feature on image2 (warped feature from image1)
    # kpts = [B*K, 2] keypoints on image2 (warped coordinates from image1)
    # geom_sqr_thresh = squre threshold of x,y coordinate distance
    # batch_inds = [B*K,] takes [0,batch_size)
    # num_kpts: [B,] tf.int32
    # outputs = min_dist, arg_min [B*K]
    N = tf.shape(feats)[0]
    batch_offsets = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts)], axis=0)
    ta_inds = tf.TensorArray(dtype=tf.int32, size=N)
    init_state = (0, ta_inds)
    condition = lambda i, _: i < batch_size

    def body(i, ta_inds):
        curr_inds = tf.cast(tf.reshape(tf.where(tf.equal(batch_inds, i)), [-1]), tf.int32)
        is_empty = tf.equal(tf.size(curr_inds), 0)
        curr_inds = tf.cond(is_empty, lambda: tf.constant([0,], dtype=tf.int32), lambda: curr_inds) # if empty use dammy index but all results are ignored
        curr_feats1 = tf.gather(feats, curr_inds) # [K,D]
        curr_feats2 = tf.gather(feats_warp, curr_inds) # [K,D]
        curr_kpts = tf.gather(kpts_warp, curr_inds) # [K,2]
        K = tf.shape(curr_feats1)[0]
        feats1_mat = tf.expand_dims(curr_feats1, axis=1) # [K,1,D]
        feats2_mat = tf.expand_dims(curr_feats2, axis=0) # [1,K,D]
        feats1_mat = tf.tile(feats1_mat, [1,K,1]) # [K,K,D]
        feats2_mat = tf.tile(feats2_mat, [K,1,1])
        feat_dists = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [K,K]
        kp1_mat = tf.expand_dims(curr_kpts, axis=1) # [K,1,2]
        kp2_mat = tf.expand_dims(curr_kpts, axis=0) # [1,K,2]
        kp1_mat = tf.tile(kp1_mat, [1,K,1]) # [K,K,2]
        kp2_mat = tf.tile(kp2_mat, [K,1,1])
        geom_dists = tf.reduce_sum(tf.squared_difference(kp1_mat, kp2_mat), axis=-1) # [K,K]
        neighbor_penalty = tf.cast(tf.less_equal(geom_dists, geom_sq_thresh), tf.float32) * 1e5
        feat_dists = feat_dists + neighbor_penalty # avoid to pickup from neighborhood

        # sort top_k and pickup from them
        topk_dist, topk_inds = tf.nn.top_k(-feat_dists, k=num_pickup, sorted=False) # take the smallest value
        # topk_dist = -topk_dist # convert comment out because dist are not necessary
        pickup_inds = tf.concat([
                        tf.range(K,dtype=tf.int32)[:,None],
                        tf.random_uniform([K,1], minval=0, maxval=num_pickup, dtype=tf.int32)
                        ], axis=1)
        min_inds = tf.gather_nd(topk_inds, pickup_inds)

        offset = batch_offsets[i]
        min_inds = min_inds + offset

        ta_inds = tf.cond(is_empty, lambda: ta_inds, lambda: ta_inds.scatter(curr_inds, min_inds))
        return i+1, ta_inds
#         return i+1, ta_dist.scatter(curr_inds, min_dist), ta_inds.scatter(curr_inds, arg_min)

    n, ta_inds_final = tf.while_loop(condition, body, init_state,
                                       parallel_iterations=num_parallel,
                                       back_prop=back_prop)
    min_inds = ta_inds_final.stack()
    return min_inds

def find_hard_negative_from_myself_with_geom_constrain_less_memory(feats, feats_warp, kpts_warp, batch_inds, num_kpts, batch_size, geom_sq_thresh, num_parallel=1, back_prop=False):
    # find nearest ref-feats index
    # feats = [B*K, D] feature on image1
    # feats_warp = [B*K, D] feature on image2 (warped feature from image1)
    # kpts = [B*K, 2] keypoints on image2 (warped coordinates from image1)
    # geom_sqr_thresh = squre threshold of x,y coordinate distance
    # batch_inds = [B*K,] takes [0,batch_size)
    # num_kpts: [B,] tf.int32
    # outputs = min_dist, arg_min [B*K]
    N = tf.shape(feats)[0]
    batch_offsets = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts)], axis=0)
    ta_dist = tf.TensorArray(dtype=tf.float32, size=N)
    ta_inds = tf.TensorArray(dtype=tf.int32, size=N)
    init_state = (0, ta_dist, ta_inds)
    condition = lambda i, _, _2: i < batch_size

    def body(i, ta_dist, ta_inds):
        curr_inds = tf.cast(tf.reshape(tf.where(tf.equal(batch_inds, i)), [-1]), tf.int32)
        is_empty = tf.equal(tf.size(curr_inds), 0)
        curr_inds = tf.cond(is_empty, lambda: tf.constant([0,], dtype=tf.int32), lambda: curr_inds) # if empty use dammy index but all results are ignored
        curr_feats1 = tf.gather(feats, curr_inds) # [K,D]
        curr_feats2 = tf.gather(feats_warp, curr_inds) # [K,D]
        curr_kpts = tf.gather(kpts_warp, curr_inds) # [K,2]
        K = tf.shape(curr_feats1)[0]
        feats1_mat = tf.expand_dims(curr_feats1, axis=1) # [K,1,D]
        feats2_mat = tf.expand_dims(curr_feats2, axis=0) # [1,K,D]
        feats1_mat = tf.tile(feats1_mat, [1,K,1]) # [K,K,D]
        feats2_mat = tf.tile(feats2_mat, [K,1,1])
        feat_dists = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [K,K]
        kp1_mat = tf.expand_dims(curr_kpts, axis=1) # [K,1,2]
        kp2_mat = tf.expand_dims(curr_kpts, axis=0) # [1,K,2]
        kp1_mat = tf.tile(kp1_mat, [1,K,1]) # [K,K,2]
        kp2_mat = tf.tile(kp2_mat, [K,1,1])
        geom_dists = tf.reduce_sum(tf.squared_difference(kp1_mat, kp2_mat), axis=-1) # [K,K]
        neighbor_penalty = tf.cast(tf.less_equal(geom_dists, geom_sq_thresh), tf.float32) * 1e5
        feat_dists = feat_dists + neighbor_penalty # avoid to pickup from neighborhood
        min_dist = tf.reduce_min(feat_dists, axis=1) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2 [B,K]
        arg_min = tf.argmin(feat_dists, axis=1, output_type=tf.int32) # find closest warped feats by using argmin(axis=1)

        offset = batch_offsets[i]
        arg_min = arg_min + offset

        ta_dist = tf.cond(is_empty, lambda: ta_dist, lambda: ta_dist.scatter(curr_inds, min_dist))
        ta_inds = tf.cond(is_empty, lambda: ta_inds, lambda: ta_inds.scatter(curr_inds, arg_min))
        return i+1, ta_dist, ta_inds
#         return i+1, ta_dist.scatter(curr_inds, min_dist), ta_inds.scatter(curr_inds, arg_min)

    n, ta_dist_final, ta_inds_final = tf.while_loop(condition, body, init_state,
                                                   parallel_iterations=num_parallel,
                                                   back_prop=back_prop)
    min_dist = ta_dist_final.stack()
    min_inds = ta_inds_final.stack()
    return min_dist, min_inds

def imperfect_find_hard_negative_from_myself_with_geom_constrain_less_memory(feats, kpts, batch_inds, num_kpts, batch_size, geom_sq_thresh, num_parallel=1, back_prop=False):
    # feats = [B*K, D]
    # kpts = [B*K, 2]
    # geom_sqr_thresh = squre threshold of x,y coordinate distance
    # batch_inds = [B*K,] takes [0,batch_size)
    # num_kpts: [B,] tf.int32
    # outputs = min_dist, arg_min [B*K]
    N = tf.shape(feats)[0]
    batch_offsets = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts)], axis=0)
    ta_dist = tf.TensorArray(dtype=tf.float32, size=N)
    ta_inds = tf.TensorArray(dtype=tf.int32, size=N)
    init_state = (0, ta_dist, ta_inds)
    condition = lambda i, _, _2: i < batch_size

    def body(i, ta_dist, ta_inds):
        curr_inds = tf.cast(tf.reshape(tf.where(tf.equal(batch_inds, i)), [-1]), tf.int32)
        is_empty = tf.equal(tf.size(curr_inds), 0)
        curr_inds = tf.cond(is_empty, lambda: tf.constant([0,], dtype=tf.int32), lambda: curr_inds) # if empty use dammy index but all results are ignored
        curr_feats = tf.gather(feats, curr_inds) # [K,D]
        curr_kpts = tf.gather(kpts, curr_inds) # [K,2]
        K = tf.shape(curr_feats)[0]
        feats1_mat = tf.expand_dims(curr_feats, axis=1) # [K,1,D]
        feats2_mat = tf.expand_dims(curr_feats, axis=0) # [1,K,D]
        feats1_mat = tf.tile(feats1_mat, [1,K,1]) # [K,K,D]
        feats2_mat = tf.tile(feats2_mat, [K,1,1])
        feat_dists = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [K,K]
        kp1_mat = tf.expand_dims(curr_kpts, axis=1) # [K,1,2]
        kp2_mat = tf.expand_dims(curr_kpts, axis=0) # [1,K,2]
        kp1_mat = tf.tile(kp1_mat, [1,K,1]) # [K,K,2]
        kp2_mat = tf.tile(kp2_mat, [K,1,1])
        geom_dists = tf.reduce_sum(tf.squared_difference(kp1_mat, kp2_mat), axis=-1) # [K,K]
        neighbor_penalty = tf.cast(tf.less_equal(geom_dists, geom_sq_thresh), tf.float32) * 1e5
        feat_dists = feat_dists + neighbor_penalty # avoid to pickup from neighborhood
        min_dist = tf.reduce_min(feat_dists, axis=1) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2 [B,K]
        arg_min = tf.argmin(feat_dists, axis=1, output_type=tf.int32)

        offset = batch_offsets[i]
        arg_min = arg_min + offset

        ta_dist = tf.cond(is_empty, lambda: ta_dist, lambda: ta_dist.scatter(curr_inds, min_dist))
        ta_inds = tf.cond(is_empty, lambda: ta_inds, lambda: ta_inds.scatter(curr_inds, arg_min))
        return i+1, ta_dist, ta_inds
#         return i+1, ta_dist.scatter(curr_inds, min_dist), ta_inds.scatter(curr_inds, arg_min)

    n, ta_dist_final, ta_inds_final = tf.while_loop(condition, body, init_state,
                                                   parallel_iterations=num_parallel,
                                                   back_prop=back_prop)
    min_dist = ta_dist_final.stack()
    min_inds = ta_inds_final.stack()
    return min_dist, min_inds

def find_random_negative_from_myself_less_memory(feats, batch_inds, num_kpts, batch_size, num_parallel=1, back_prop=False):
    # feats = [B*K, D]
    # batch_inds = [B*K,] takes [0,batch_size)
    # num_kpts: [B,] tf.int32
    # outputs = min_dist, arg_min [B*K]
    N = tf.shape(feats)[0]
    batch_offsets = tf.concat([tf.zeros(1, dtype=tf.int32), tf.cumsum(num_kpts)], axis=0)
    ta_inds = tf.TensorArray(dtype=tf.int32, size=N)
    init_state = (0, ta_inds)
    condition = lambda i, _,: i < batch_size

    def body(i, ta_inds):
        curr_inds = tf.cast(tf.reshape(tf.where(tf.equal(batch_inds, i)), [-1]), tf.int32)
        is_empty = tf.equal(tf.size(curr_inds), 0)
        curr_inds = tf.cond(is_empty, lambda: tf.constant([0,], dtype=tf.int32), lambda: curr_inds) # if empty use dammy index but all results are ignored
        curr_feats = tf.gather(feats, curr_inds) # [K,D]
        K = tf.shape(curr_feats)[0]
        # could be select itself in case K=1
        rnd_inds = tf.random_uniform([K], minval=1, maxval=tf.maximum(K,2), dtype=tf.int32) # random_uniform doesn't allow the case minval==maxval
#         rnd_inds = tf.ones([K], dtype=tf.int32) * 2
        rnd_inds = tf.range(K) + rnd_inds
        rnd_inds = tf.where(tf.less(rnd_inds, K), rnd_inds, rnd_inds-K)
        rnd_inds = rnd_inds + batch_offsets[i]

        ta_inds = tf.cond(is_empty, lambda: ta_inds, lambda: ta_inds.scatter(curr_inds, rnd_inds))
        return i+1, ta_inds

    n, ta_inds_final = tf.while_loop(condition, body, init_state,
                                    parallel_iterations=num_parallel,
                                    back_prop=back_prop)
    rnd_inds = ta_inds_final.stack()
    return rnd_inds


# def find_correspondences(points, indices):
#     # points: tf.float32, [B,K,2]
#     # indices: tf.int32, [B,L] indices[i,j] = [0,K)
#     B = tf.shape(indices)[0]
#     L = tf.shape(indices)[1]

#     batch_indices = tf.tile(tf.range(B, dtype=indices.dtype)[:,None], [1,L])
#     indices = tf.reshape(indices, [-1,1])
#     batch_indices = tf.reshape(batch_indices, [-1,1])
#     indices = tf.concat([batch_indices, indices], axis=1) # [B*L,2]

#     gathered_points = tf.gather_nd(points, indices)
#     gathered_points = tf.reshape(gathered_points, [B,L,-1])
#     return gathered_points

# def batch_nearest_neighbors(feats1, feats2):
#     # feats1.shape = [B,K,D]
#     # feats2.shape = [B,L,D]
#     K = tf.shape(feats1)[1]
#     L = tf.shape(feats2)[1]

#     feats1_mat = tf.expand_dims(feats1, axis=2) # [B,K,D] --> [B,K,1,D]
#     feats2_mat = tf.expand_dims(feats2, axis=1) # [B,L,D] --> [B,1,L,D]
#     feats1_mat = tf.tile(feats1_mat, [1,1,L,1]) # [B,K,L,D]
#     feats2_mat = tf.tile(feats2_mat, [1,K,1,1]) # [B,K,L,D]

#     distances = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1) # [B,K,L]
#     min_dist1 = tf.reduce_min(distances, axis=2) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2 [B,K]
#     arg_min1 = tf.argmin(distances, axis=2, output_type=tf.int32)
#     min_dist2 = tf.reduce_min(distances, axis=1) # [B,L]
#     arg_min2 = tf.argmin(distances, axis=1, output_type=tf.int32)

#     return min_dist1, arg_min1, min_dist2, arg_min2

def nearest_neighbors(feats1, feats2):
    # feats1.shape = [K,D]
    # feats2.shape = [L,D]
    K = tf.shape(feats1)[0]
    L = tf.shape(feats2)[0]

    feats1_mat = tf.expand_dims(feats1, axis=1) # [K,D] --> [K,1,D]
    feats2_mat = tf.expand_dims(feats2, axis=0) # [K,D] --> [1,L,D]
    feats1_mat = tf.tile(feats1_mat, [1,L,1]) # [K,L,D]
    feats2_mat = tf.tile(feats2_mat, [K,1,1]) # [K,L,D]

    distances = tf.reduce_sum(tf.squared_difference(feats1_mat, feats2_mat), axis=-1)
    min_dist1 = tf.reduce_min(distances, axis=1) # min_dist1(i) = min_j |feats1(i)- feats2(j)|^2
    arg_min1 = tf.argmin(distances, axis=1)
    min_dist2 = tf.reduce_min(distances, axis=0)
    arg_min2 = tf.argmin(distances, axis=0)

    return min_dist1, arg_min1, min_dist2, arg_min2, distances


# def nearest_neighbors_less_memory(feats1, feats2, back_prop=False, num_parallel=10):
#     # min_dist(i) = min_j |feats1(i)-feats(j|**2
#     # arg_min(i) = arg min_j |feats1(i)-feats(j|**2
#     # Less memory but too slow on notebook but why ?
#     N = tf.shape(feats1)[0]
#     ta_dist = tf.TensorArray(dtype=tf.float32, size=N)
#     ta_inds = tf.TensorArray(dtype=tf.int32, size=N)

#     init_state = (0, ta_dist, ta_inds)
#     condition = lambda i, _, _2: i < N

#     def body(i, ta_dist, ta_inds):
#         dists = tf.reduce_sum( tf.squared_difference(feats1[i], feats2), axis=1)
#         min_dist = tf.reduce_min(dists)
#         min_inds = tf.cast(tf.argmin(dists), tf.int32)
#         return i+1, ta_dist.write(i, min_dist), ta_inds.write(i, min_inds)

#     n, ta_dist_final, ta_inds_final = tf.while_loop(condition, body, init_state,
#                                         parallel_iterations=num_parallel, back_prop=back_prop)

#     min_dist = ta_dist_final.stack()
#     min_inds = ta_inds_final.stack()

#     return min_dist, min_inds

def extract_keypoints(top_k):
    coords = tf.where(tf.greater(top_k, 0.))
    num_kpts = tf.reduce_sum(top_k, axis=[1,2,3])
    coords = tf.cast(coords, tf.int32)
    with tf.device('/cpu:0'):
        batch_inds, kp_y, kp_x, _ = tf.split(coords, 4, axis=-1)
    batch_inds = tf.reshape(batch_inds, [-1])
    kpts = tf.concat([kp_x, kp_y], axis=1)

    num_kpts = tf.cast(num_kpts, tf.int32)
    # kpts: [N,2] (N=B*K)
    # batch_inds: N,
    # num_kpts: B
    return kpts, batch_inds, num_kpts

def extract_patches_from_keypoints(feat_maps, kpts, batch_inds, crop_radius, patch_size):
    # feat_maps: [B,H,W,C]
    # kpts: [N,2]
    # batch_inds: [N,]
    # crop_radius, patch_size: scalar

    patch_size = (patch_size, patch_size)
    with tf.name_scope('extract_patches_from_keypoints'):
        kp_x, kp_y = tf.split(kpts, 2, axis=-1)

        bboxes = tf.cast(tf.concat([kp_y-crop_radius, kp_x-crop_radius, kp_y+crop_radius, kp_x+crop_radius],
                            axis=1), tf.float32)# [num_boxes, 4]
        # normalize bounding boxes
        height = tf.cast(tf.shape(feat_maps)[1], tf.float32)
        width = tf.cast(tf.shape(feat_maps)[2], tf.float32)
        inv_imsizes = tf.stack([1./height, 1./width, 1./height, 1./width])[None] # [1,4]
        bboxes = bboxes * inv_imsizes

        crop_patches = tf.image.crop_and_resize(feat_maps, bboxes, batch_inds, patch_size)


        return crop_patches

# def extract_patches_from_keypoints(feat_maps, top_k, crop_radius, patch_size):
#     patch_size = (patch_size, patch_size)
#     with tf.name_scope('extract_patches_from_keypoints'):
#         coords = tf.where(tf.greater(top_k, 0.))
#         coords = tf.cast(coords, tf.int32)
#         box_ind, kp_y, kp_x, _ = tf.split(coords, 4, axis=-1)
#         box_ind = tf.reshape(box_ind, [-1])

#         kpts = tf.concat([kp_x, kp_y], axis=1)

#         bboxes = tf.cast(tf.concat([kp_y-crop_radius, kp_x-crop_radius, kp_y+crop_radius, kp_x+crop_radius],
#                             axis=1), tf.float32)# [num_boxes, 4]
#         # normalize bounding boxes
#         height = tf.cast(tf.shape(feat_maps)[1], tf.float32)
#         width = tf.cast(tf.shape(feat_maps)[2], tf.float32)
#         inv_imsizes = tf.stack([1./height, 1./width, 1./height, 1./width])[None] # [1,4]
#         bboxes = bboxes * inv_imsizes

#         crop_patches = tf.image.crop_and_resize(feat_maps, bboxes, box_ind, patch_size)


#         return crop_patches, kpts, box_ind

def make_intrinsics_3x3(fx, fy, cx, cy):
    # non-batch inputs, outputs
    # inputs are scalar, output is 3x3 tensor
    r1 = tf.expand_dims(tf.stack([fx, 0., cx]), axis=0) # [1,3]
    r2 = tf.expand_dims(tf.stack([0., fy, cy]), axis=0)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    intrinsics = tf.concat([r1, r2, r3], axis=0)
    return intrinsics

def get_gauss_filter_weight(ksize, sig):
    mu_x = mu_y = ksize//2
    if sig == 0:
        psf = np.zeros((ksize, ksize), dtype=np.float32)
        psf[mu_y,mu_x] = 1.0
    else:
        xy = np.indices((ksize,ksize))
        x = xy[1,:,:]
        y = xy[0,:,:]
        psf  = np.exp(-((x-mu_x)**2/(2*sig**2) + (y-mu_y)**2/(2*sig**2)))
    return psf

def spatial_softmax(logits, ksize, com_strength=1.0):

    max_logits = tf.nn.max_pool(logits, [1,ksize,ksize, 1],
                            strides=[1,1,1,1], padding='SAME')
    sum_filter = tf.constant(np.ones((ksize,ksize,1,1)), dtype=tf.float32)
    ex = tf.exp(com_strength * (logits - max_logits))
    sum_ex = tf.nn.conv2d(ex, sum_filter, [1,1,1,1], padding='SAME')
    probs = ex / (sum_ex + 1e-6)
    return probs

def soft_nms_3d(scale_logits, ksize, com_strength=1.0):
    # apply softmax on scalespace logits
    # scale_logits: [B,H,W,S]
    num_scales = scale_logits.get_shape().as_list()[-1]

    scale_logits_d = tf.transpose(scale_logits[...,None], [0,3,1,2,4]) # [B,S,H,W,1] in order to apply pool3d
    max_maps = tf.nn.max_pool3d(scale_logits_d, [1,num_scales,ksize,ksize,1], [1,num_scales,1,1,1], padding='SAME')
    max_maps = tf.transpose(max_maps[...,0], [0,2,3,1]) # [B,H,W,S]
    exp_maps = tf.exp(com_strength * (scale_logits-max_maps))
    exp_maps_d = tf.transpose(exp_maps[...,None], [0,3,1,2,4]) # [B,S,H,W,1]
    sum_filter = tf.constant(np.ones((num_scales, ksize, ksize, 1, 1)), dtype=tf.float32)
    sum_ex = tf.nn.conv3d(exp_maps_d, sum_filter, [1,num_scales,1,1,1], padding='SAME')
    sum_ex = tf.transpose(sum_ex[...,0], [0,2,3,1]) # [B,H,W,S]
    probs = exp_maps / (sum_ex + 1e-6)

    return probs

def instance_normalization(inputs):
    # normalize 0-means 1-variance in each sample (not take batch-axis)
    inputs_dim = inputs.get_shape().ndims
    # Epsilon to be used in the tf.nn.batch_normalization
    var_eps = 1e-3
    if inputs_dim == 4:
        moments_dims = [1,2] # NHWC format
    elif inputs_dim == 2:
        moments_dims = [1]
    else:
        raise ValueError('instance_normalization suppose input dim is 4: inputs_dim={}\n'.format(inputs_dim))

    mean, variance = tf.nn.moments(inputs, axes=moments_dims, keep_dims=True)
    outputs = tf.nn.batch_normalization(inputs, mean, variance, None, None, var_eps) # non-parametric normalization
    return outputs


def non_max_suppression(inputs, thresh=0.0, ksize=3, dtype=tf.float32, name='NMS'):
    with tf.name_scope(name): # add namespace to keep graph clean
        dtype = inputs.dtype
        batch = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channel = tf.shape(inputs)[3]
        hk = ksize // 2
        zeros = tf.zeros_like(inputs)
        works = tf.where(tf.less(inputs, thresh), zeros, inputs)
        works_pad = tf.pad(works, [[0,0], [2*hk,2*hk], [2*hk,2*hk], [0,0]], mode='CONSTANT')
        map_augs = []
        for i in range(ksize):
            for j in range(ksize):
                curr_in = tf.slice(works_pad, [0, i, j, 0], [-1, height+2*hk, width+2*hk, -1])
                map_augs.append(curr_in)

        num_map = len(map_augs) # ksize*ksize
        center_map = map_augs[num_map//2]
        peak_mask = tf.greater(center_map, map_augs[0])
        for n in range(1, num_map):
            if n == num_map // 2:
                continue
            peak_mask = tf.logical_and(peak_mask, tf.greater(center_map, map_augs[n]))
        with tf.device('/cpu:0'):
            peak_mask = tf.slice(peak_mask, [0,hk,hk,0],[-1,height,width,-1])
        if dtype != tf.bool:
            peak_mask = tf.cast(peak_mask, dtype=dtype)
        peak_mask.set_shape(inputs.shape) # keep shape information
        return peak_mask

# def make_top_k_sparse_tensor(heatmaps, k=256):
#     batch_size = tf.shape(heatmaps)[0]
#     _, height, width, channel = heatmaps.get_shape().as_list()
#     heatmaps_flt = tf.reshape(heatmaps, [batch_size, -1])
#     values, indices = tf.nn.top_k(heatmaps_flt, k=k, sorted=False)
#     boffset = tf.expand_dims(tf.range(batch_size) * width * height, axis=1)
#     indices = indices + boffset
#     indices = tf.reshape(indices, [-1])
#     top_k_maps = tf.sparse_to_dense(indices, [batch_size*height*width*channel], 1, 0, validate_indices=False)
#     top_k_maps = tf.reshape(top_k_maps, [batch_size, height, width, 1])
#     return tf.cast(top_k_maps, tf.float32)

def make_top_k_sparse_tensor(heatmaps, k=256, get_kpts=False):
    batch_size = tf.shape(heatmaps)[0]
    height = tf.shape(heatmaps)[1]
    width = tf.shape(heatmaps)[2]
    heatmaps_flt = tf.reshape(heatmaps, [batch_size, -1])
    imsize = tf.shape(heatmaps_flt)[1]

    values, xy_indices = tf.nn.top_k(heatmaps_flt, k=k, sorted=False)
    boffset = tf.expand_dims(tf.range(batch_size) * imsize, axis=1)
    indices = xy_indices + boffset
    indices = tf.reshape(indices, [-1])
    with tf.device('/cpu:0'):
        top_k_maps = tf.sparse_to_dense(indices, [batch_size*imsize], 1, 0, validate_indices=False)
    top_k_maps = tf.reshape(top_k_maps, [batch_size, height, width, 1])
    top_k_maps = tf.cast(top_k_maps, tf.float32)
    if get_kpts:
        kpx = tf.mod(xy_indices, width)
        kpy = xy_indices // width
        batch_inds = tf.tile(tf.range(batch_size, dtype=tf.int32)[:,None], [1,k])
        kpts = tf.concat([tf.reshape(kpx, [-1,1]), tf.reshape(kpy, [-1,1])], axis=1) # B*K,2
        batch_inds = tf.reshape(batch_inds, [-1])
        num_kpts = tf.ones([batch_size], dtype=tf.int32) * k
        return top_k_maps, kpts, batch_inds, num_kpts
    else:
        return top_k_maps


def d_softargmax(d_heatmaps, block_size, com_strength=10):
    # d_heatmaps = [batch, height/N, width/N, N**2]
    # fgmask = [batch, height/N, width/N]
    pos_array_x, pos_array_y = tf.meshgrid(tf.range(block_size), tf.range(block_size))
    pos_array_x = tf.cast(tf.reshape(pos_array_x, [-1]), tf.float32)
    pos_array_y = tf.cast(tf.reshape(pos_array_y, [-1]), tf.float32)

    max_out = tf.reduce_max(
        d_heatmaps, axis=-1, keep_dims=True)
    o = tf.exp(com_strength * (d_heatmaps - max_out))  # + eps
    sum_o = tf.reduce_sum(
        o, axis=-1, keep_dims=True)
    x = tf.reduce_sum(
        o * tf.reshape(pos_array_x, [1, 1, 1, -1]),
        axis=-1, keep_dims=True
    ) / sum_o
    y = tf.reduce_sum(
        o * tf.reshape(pos_array_y, [1, 1, 1, -1]),
        axis=-1, keep_dims=True
    ) / sum_o

    # x,y shape = [B,H,W,1]
    coords = tf.concat([x, y], axis=-1)
    # x = tf.squeeze(x, axis=-1)
    # y = tf.squeeze(y, axis=-1)

    return coords

def softargmax(score_map, com_strength=10):
    # out.shape = [batch, height, width, 1]
    height = tf.shape(score_map)[1]
    width = tf.shape(score_map)[2]
    md = len(score_map.shape)
    # CoM to get the coordinates
    pos_array_x = tf.cast(tf.range(height), dtype=tf.float32)
    pos_array_y = tf.cast(tf.range(height), dtype=tf.float32)

    max_out = tf.reduce_max(
        score_map, axis=list(range(1, md)), keep_dims=True)
    o = tf.exp(com_strength * (score_map - max_out))  # + eps
    sum_o = tf.reduce_sum(
        o, axis=list(range(1, md)), keep_dims=True)
    x = tf.reduce_sum(
        o * tf.reshape(pos_array_x, [1, 1, -1, 1]),
        axis=list(range(1, md)), keep_dims=True
    ) / sum_o
    y = tf.reduce_sum(
        o * tf.reshape(pos_array_y, [1, -1, 1, 1]),
        axis=list(range(1, md)), keep_dims=True
    ) / sum_o

    # Remove the unecessary dimensions (i.e. flatten them)
    x = tf.reshape(x, (-1,))
    y = tf.reshape(y, (-1,))

    return x, y

##--------------------------
## Bilinear Inverse Warping
##--------------------------

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords

# def pixel2cam(depths, pixel_coords, inv_intrinsics_3x3, is_homogeneous=True):
#     """Transforms coordinates in the pixel frame to the camera frame.

#     Args:
#         depths: [batch, height, width]
#         pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
#         intrinsics: camera intrinsics [batch, 3, 3]
#         is_homogeneous: return in homogeneous coordinates
#     Returns:
#         Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
#     """
#     batch = tf.shape(depths)[0]
#     height = tf.shape(depths)[1]
#     width = tf.shape(depths)[2]

#     depths = tf.reshape(depths, [batch, 1, -1])
#     pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
#     inv_intrinsics = tf.tile(tf.expand_dims(inv_intrinsics_3x3, axis=0), [batch, 1, 1])
#     cam_coords = tf.matmul(inv_intrinsics, pixel_coords) * depths
#     if is_homogeneous:
#         ones = tf.ones([batch, 1, height*width])
#         cam_coords = tf.concat([cam_coords, ones], axis=1)
#     cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
#     return cam_coords
def pixel2cam(depths, pixel_coords, inv_intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
        depths: [batch, height, width]
        pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
        intrinsics: camera intrinsics [batch, 3, 3]
        is_homogeneous: return in homogeneous coordinates
    Returns:
        Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch = tf.shape(depths)[0]
    height = tf.shape(depths)[1]
    width = tf.shape(depths)[2]

    depths = tf.reshape(depths, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(inv_intrinsics, pixel_coords) * depths

    # if thetas is not None:
    #     inv_thetas = tf.matrix_inverse(thetas)
    #     cam_coords = tf.matmul(inv_thetas, cam_coords)
    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
        cam_coords: [batch, 4, height, width]
        proj: [batch, 4, 4]
    Returns:
        Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch = tf.shape(cam_coords)[0]
    height = tf.shape(cam_coords)[2]
    width = tf.shape(cam_coords)[3]

    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    reproj_depth = tf.reshape(z_u, [batch, 1, height, width])
    reproj_depth = tf.transpose(reproj_depth, perm=[0,2,3,1])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1]), reproj_depth

def get_visibility_mask(coords):
    """ Get visible region mask
    Args:
        coords: [batch, height, width, 2]
    Return:
        visible mask [batch, height, width, 1]
    """

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.cast(tf.floor(coords_x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(coords_y), tf.int32)
    y1 = y0 + 1

    height = tf.shape(coords)[1]
    width = tf.shape(coords)[2]
    zero = tf.zeros([1], dtype=tf.int32)
    inside_x = tf.logical_and(tf.greater_equal(x0, zero), tf.less(x1, width))
    inside_y = tf.logical_and(tf.greater_equal(y0, zero), tf.less(y1, height))
    visible_mask = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32)

    return visible_mask

def bilinear_sampling(photos, coords, depths=None):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
        photos: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
          width_t, 2]. height_t/width_t correspond to the dimensions of the output
          image (don't need to be the same as height_s/width_s). The two channels
          correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """
    # photos: [batch_size, height2, width2, C]
    # coords: [batch_size, height1, width1, C]
    # depths: [batch_size, height2, width2, 1]
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = tf.shape(photos)
        if depths is not None:
            dpt_size = tf.shape(depths)
        else:
            dpt_size = inp_size # dammy
        coord_size = tf.shape(coords)

        out_size = tf.stack([coord_size[0],
                             coord_size[1],
                             coord_size[2],
                             inp_size[3],
                            ])
        out_size_d = tf.stack([coord_size[0],
                             coord_size[1],
                             coord_size[2],
                             dpt_size[3],
                            ])

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(photos)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(photos)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1


        ## sample from photos
        photos_flat = tf.reshape(photos, tf.stack([-1, inp_size[3]]))
        photos_flat = tf.cast(photos_flat, 'float32')

        im00 = tf.reshape(tf.gather(photos_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(photos_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(photos_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(photos_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        out_photos = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        if depths is None:
            return out_photos
        else:
            ## sample from depths
            dpts_flat = tf.reshape(depths, tf.stack([-1, dpt_size[3]]))
            dpts_flat = tf.cast(dpts_flat, 'float32')

            dp00 = tf.reshape(tf.gather(dpts_flat, tf.cast(idx00, 'int32')), out_size_d)
            dp01 = tf.reshape(tf.gather(dpts_flat, tf.cast(idx01, 'int32')), out_size_d)
            dp10 = tf.reshape(tf.gather(dpts_flat, tf.cast(idx10, 'int32')), out_size_d)
            dp11 = tf.reshape(tf.gather(dpts_flat, tf.cast(idx11, 'int32')), out_size_d)

            out_depth = tf.add_n([
                w00 * dp00, w01 * dp01,
                w10 * dp10, w11 * dp11
            ])

            return out_photos, out_depth

def nearest_neighbor_sampling(photos, coords):
    # photos: [batch_size, height2, width2, C]
    # coords: [batch_size, height1, width1, C]
    # outputs: [batch_size, height1, width1, C]

    inp_size = tf.shape(photos)
    coord_size = tf.shape(coords)
    out_size = tf.stack([coord_size[0],
                         coord_size[1],
                         coord_size[2],
                         inp_size[3],
                        ])
    batch_size = inp_size[0]
    in_height = inp_size[1]
    in_width = inp_size[2]
    out_height = out_size[1]
    out_width = out_size[2]

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)

    # [height+2,width+2]
    photos_pad = tf.pad(photos, [[0, 0], [1, 1], [1, 1], [0,0]], 'CONSTANT')

    # valid area = 1 <= x,y <= width
    coords_x_pad = tf.clip_by_value(tf.cast(tf.round(coords_x)+1, tf.int32), 0, in_width+1)
    coords_y_pad = tf.clip_by_value(tf.cast(tf.round(coords_y)+1, tf.int32), 0, in_height+1)

    batch_inds = tf.tile(tf.range(batch_size)[:,None,None,None], [1, out_height, out_width,1])
    byx = tf.concat([batch_inds, coords_y_pad, coords_x_pad], axis=-1)

    outputs = tf.gather_nd(photos_pad, byx)
    return outputs

# def projective_inverse_warp(photos1, depths2, c1Tc2s, intrinsics_3x3, depths1, depth_thresh=1.0, name='InvWarp'):
#     '''
#     Args:
#         photos1 : [B,H,W,C] photos of camera1
#         depths2 : [B,H,W,C] depths of camera2
#         c1Tc2s : [B,4,4] pose matrix from camera2 to camera1
#         intrinsics_3x3 : [3,3] intrinsic matrix
#         depths1 : [B,H,W,C] (option) depths of camera1. depths1 is used to remove occlusion from valid_masks
#         depth_thresh : float (option) threshold for depth misconsistency between depths2 and warped depths
#     '''
#     with tf.name_scope(name):
#         batch = tf.shape(photos1)[0]
#         height = tf.shape(photos1)[1]
#         width = tf.shape(photos1)[2]

#         inv_intrinsics_3x3 = tf.matrix_inverse(intrinsics_3x3)
#         batch_intrinsics_4x4 = tf.concat([intrinsics_3x3, tf.zeros([3,1])], axis=1)
#         batch_intrinsics_4x4 = tf.concat([batch_intrinsics_4x4, tf.constant([0.0,0.0,0.0,1.0], shape=[1,4])], axis=0)
#         batch_intrinsics_4x4 = tf.expand_dims(batch_intrinsics_4x4, axis=0) # [1,4,4]
#         batch_intrinsics_4x4 = tf.tile(batch_intrinsics_4x4, [batch, 1, 1]) # [B,4,4]

#         pixel_coords_c2 = meshgrid(batch, height, width)
#         cam_coords_c2 = pixel2cam(depths2, pixel_coords_c2, inv_intrinsics_3x3)
#         proj_c1Tc2s = tf.matmul(batch_intrinsics_4x4, c1Tc2s)
#         pixel_coords_c1, reproj_depths = cam2pixel(cam_coords_c2, proj_c1Tc2s)

#         warp_photos, warp_depths = bilinear_sampling(photos1, pixel_coords_c1, depths1)

#         warp_photos.set_shape(photos1.shape)
#         warp_depths.set_shape(depths1.shape)

#         # visiblity mask
#         visible_masks = get_visibility_mask(pixel_coords_c1)
#         camfront_masks = tf.cast(tf.greater(reproj_depths, tf.zeros((), reproj_depths.dtype)), tf.float32)
#         nonocc_masks = tf.cast(tf.less(tf.squared_difference(warp_depths, depths2), depth_thresh**2), tf.float32)
#         visible_masks = visible_masks * camfront_masks * nonocc_masks # take logical_and
#         warp_photos = warp_photos * visible_masks

#     return warp_photos, visible_masks

def projective_inverse_warp(photos1, depths2, c1Tc2s, intrinsics_3x3, depths1, depth_thresh=1.0, name='InvWarp'):
    '''
    Args:
        photos1 : [B,H,W,C] photos of camera1
        depths2 : [B,H,W,C] depths of camera2
        c1Tc2s : [B,4,4] pose matrix from camera2 to camera1
        intrinsics_3x3 : [B,3,3] intrinsic matrix
        depths1 : [B,H,W,C] (option) depths of camera1. depths1 is used to remove occlusion from valid_masks
        depth_thresh : float (option) threshold for depth misconsistency between depths2 and warped depths
    '''
    with tf.name_scope(name):
        batch = tf.shape(photos1)[0]
        height = tf.shape(photos1)[1]
        width = tf.shape(photos1)[2]

        inv_intrinsics_3x3 = tf.matrix_inverse(intrinsics_3x3)


        # if thetas1 is not None:
        #     inv_thetas1 = tf.matrix_inverse(thetas1)
        #     inv_intrinsics_3x3 = tf.matmul(inv_thetas1, inv_intrinsics_3x3)

        right_col = tf.zeros([batch,3,1], dtype=tf.float32)
        bottom_row = tf.tile(tf.constant([0,0,0,1], dtype=tf.float32, shape=[1,1,4]), [batch,1,1])
        batch_intrinsics_4x4 = tf.concat([intrinsics_3x3, right_col], axis=2)
        batch_intrinsics_4x4 = tf.concat([batch_intrinsics_4x4, bottom_row], axis=1)

        pixel_coords_c2 = meshgrid(batch, height, width)
        cam_coords_c2 = pixel2cam(depths2, pixel_coords_c2, inv_intrinsics_3x3)
        proj_c1Tc2s = tf.matmul(batch_intrinsics_4x4, c1Tc2s)
        pixel_coords_c1, reproj_depths = cam2pixel(cam_coords_c2, proj_c1Tc2s)

        warp_photos, warp_depths = bilinear_sampling(photos1, pixel_coords_c1, depths1)

        warp_photos.set_shape(photos1.shape)
        warp_depths.set_shape(depths1.shape)

        # visiblity mask
        visible_masks = get_visibility_mask(pixel_coords_c1)
        camfront_masks = tf.cast(tf.greater(reproj_depths, tf.zeros((), reproj_depths.dtype)), tf.float32)
        nonocc_masks = tf.cast(tf.less(tf.squared_difference(warp_depths, depths2), depth_thresh**2), tf.float32)
        visible_masks = visible_masks * camfront_masks * nonocc_masks # take logical_and
        warp_photos = warp_photos * visible_masks

    return warp_photos, visible_masks
def extract_xy_coords(d_heatmaps, block_size):
    batch = tf.shape(d_heatmaps)[0]
    rheight = tf.shape(d_heatmaps)[1]
    rwidth = tf.shape(d_heatmaps)[2]
    width = rwidth * block_size
    height = rheight * block_size

    d_argmax = tf.cast(tf.argmax(d_heatmaps, axis=-1), dtype=tf.int32)
    fgmask = tf.cast(tf.not_equal(d_argmax, block_size**2), dtype=tf.int32)

    x_bcoords = tf.mod(d_argmax, block_size)
    y_bcoords = tf.floordiv(d_argmax, block_size) # floor_div ?
    zero = tf.constant(0, dtype=tf.int32)
    zeros = tf.zeros_like(x_bcoords)

    x_bcoords = tf.where(tf.equal(fgmask, zero), zeros, x_bcoords)
    y_bcoords = tf.where(tf.equal(fgmask, zero), zeros, y_bcoords)

    x_offset, y_offset = tf.meshgrid(tf.range(0, width, block_size), tf.range(0, height, block_size))
    x_offset = tf.tile(tf.expand_dims(x_offset, axis=0), [batch, 1, 1])
    y_offset = tf.tile(tf.expand_dims(y_offset, axis=0), [batch, 1, 1])

    x_icoords = x_bcoords + x_offset
    y_icoords = y_bcoords + y_offset

    return x_icoords, y_icoords, fgmask


def heatmaps_to_reprojected_heatmaps(d_heatmaps1, depths1, depths2, c2Tc1s, intrinsics, block_size, depth_thresh=1.0, name='HM2HM'):

    '''
    convert CNN output d_heatmaps to reprojected heatmaps

    Args:
        d_heatmaps1: [batch, height/N, width/N, N*N+1], CNN output depth-wise heatmaps
        depths1: [batch, height, width, 1], depths of photos1
        depths2: [batch, height, width, 1], depths of photos2 (to check depth consistency)
        c2Tc1s: [batch, 4, 4], transformation matrix from camera1 to camera2
        intrinsics: [batch, 3, 3], intrinsics matrix
        block_size: N, downsampling rate (2**#pooling)
    Return:
        heatmaps2: [batch, height, width, 1]
    '''

    with tf.name_scope(name):
        batch = tf.shape(depths1)[0]
        height = tf.shape(depths1)[1]
        width = tf.shape(depths1)[2]

        rheight = tf.shape(d_heatmaps1)[1]
        rwidth = tf.shape(d_heatmaps1)[2]

        # extract keypoints from each blocks
        with tf.name_scope('KP-EXTRACT'):
            x1, y1, fgmask = extract_xy_coords(d_heatmaps1, block_size)
            fgmask = tf.reshape(fgmask, (batch, 1, -1))
        with tf.name_scope('XY2IDX'):
            # convert (x,y) to idx=x+y*width+b*width*height
            x1 = tf.reshape(x1, (batch, 1, -1))
            y1 = tf.reshape(y1, (batch, 1, -1))
            b_offset = tf.range(batch) * width * height
            b_offset = tf.tile(tf.expand_dims(b_offset, axis=1), [1, rwidth*rheight])
            b_offset = tf.reshape(b_offset, (batch, 1, -1))
            idx1 = x1 + y1 * width + b_offset
            # extract depth on (x,y)
            depth1s_flt = tf.reshape(depths1, [-1])
            z1 = tf.gather(depth1s_flt, idx1)

        with tf.name_scope('PIX2CAM'):

            # pixel to camera coordinate (x,y,1 --> x,y,z,1)
            ones = tf.ones_like(x1)
            pix_coords = tf.concat([x1, y1, ones], axis=1)
            pix_coords = tf.cast(pix_coords, tf.float32)

            inv_intrinsics = tf.matrix_inverse(intrinsics)
            cam_coords = tf.matmul(inv_intrinsics, pix_coords) * z1
            cam_coords = tf.concat([cam_coords, tf.cast(ones,tf.float32)], axis=1)

        with tf.name_scope('TRANS-MAT'):
            # get Transform matrix
            filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
            filler = tf.tile(filler, [batch, 1, 1])
            intrinsics_4x4 = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
            intrinsics_4x4 = tf.concat([intrinsics_4x4, filler], axis=1)
            proj_mats = tf.matmul(intrinsics_4x4, c2Tc1s)

        with tf.name_scope('CAM2PIX'):
            # camera to pixel coordinate
            unnormalized_pixel_coords = tf.matmul(proj_mats, cam_coords) # [batch, 4, N(=rwidth*rheight)]

            x_u = tf.slice(unnormalized_pixel_coords, [0,0,0], [-1,1,-1])
            y_u = tf.slice(unnormalized_pixel_coords, [0,1,0], [-1,1,-1])
            z_u = tf.slice(unnormalized_pixel_coords, [0,2,0], [-1,1,-1])
            # remove coordinate behind camera
            cam_front = tf.greater(z_u, 0)
            fgmask = fgmask * tf.cast(cam_front, tf.int32)
            epsilon = tf.constant(1e-6, dtype=tf.float32)
            x_n = x_u / (z_u + epsilon)
            y_n = y_u / (z_u + epsilon)

        with tf.name_scope('DepthCheck'):
            x2 = tf.clip_by_value(tf.cast(tf.round(x_n), tf.int32), 0, width-1)
            y2 = tf.clip_by_value(tf.cast(tf.round(y_n), tf.int32), 0, height-1)
            depth2s_flt = tf.reshape(depths2, [-1])
            idx2 = x2 + y2 * width + b_offset
            z2 = tf.gather(depth2s_flt, idx2)
            near_depth = tf.cast(tf.less_equal(tf.abs(z_u-z2), depth_thresh), tf.int32)
            fgmask = fgmask * near_depth

        with tf.name_scope('SUBSAMPLING'):
            # clipping coordinates
            pad = 1
            width2 = width + 2 * pad
            height2 = height + 2 * pad

            x2 = tf.clip_by_value(tf.cast(tf.round(x_n), tf.int32)+pad, 0, width+1) # 0~width+1 (actual range=1~width-1)
            y2 = tf.clip_by_value(tf.cast(tf.round(y_n), tf.int32)+pad, 0, height+1)
            zero = tf.constant(0, dtype=tf.int32)
            zeros = tf.zeros_like(x2)
            x2 = tf.where(tf.equal(fgmask, zero), zeros, x2)
            y2 = tf.where(tf.equal(fgmask, zero), zeros, y2)
            x2 = tf.reshape(x2, [batch, -1])
            y2 = tf.reshape(y2, [batch, -1])

            b_offset = tf.range(batch) * width2 * height2
            b_offset = tf.tile(tf.expand_dims(b_offset, axis=1), [1, rwidth*rheight])

            idx2 = x2 + y2 * width2 + b_offset
            idx2 = tf.reshape(idx2, [-1])

            idx2, _ = tf.unique(idx2) # remove duplicate indices to apply sparce_to_dense

            # ignore sorted order (https://stackoverflow.com/questions/35508894/sparse-to-dense-requires-indices-to-be-lexicographically-sorted-in-0-7)
            heatmaps2 = tf.sparse_to_dense(idx2, [batch*height2*width2], 1, 0, validate_indices=False)

            heatmaps2 = tf.reshape(heatmaps2, (batch, height2, width2))
            heatmaps2 = tf.slice(heatmaps2, [0,1,1], [-1,height,width])
            heatmaps2 = tf.cast(tf.expand_dims(heatmaps2, axis=-1), tf.float32)

    return heatmaps2


# def extract_xy_coords(d_heatmaps, block_size):
#     batch = tf.shape(d_heatmaps)[0]
#     rheight = tf.shape(d_heatmaps)[1]
#     rwidth = tf.shape(d_heatmaps)[2]
#     width = rwidth * block_size
#     height = rheight * block_size

#     d_argmax = tf.cast(tf.argmax(d_heatmaps, axis=-1), dtype=tf.int32)
#     fgmask = tf.cast(tf.not_equal(d_argmax, block_size**2), dtype=tf.int32)

#     x_bcoords = tf.mod(d_argmax, block_size)
#     y_bcoords = tf.floordiv(d_argmax, block_size) # floor_div ?
#     zero = tf.constant(0, dtype=tf.int32)
#     zeros = tf.zeros_like(x_bcoords)

#     x_bcoords = tf.where(tf.equal(fgmask, zero), zeros, x_bcoords)
#     y_bcoords = tf.where(tf.equal(fgmask, zero), zeros, y_bcoords)

#     x_offset, y_offset = tf.meshgrid(tf.range(0, width, block_size), tf.range(0, height, block_size))
#     x_offset = tf.tile(tf.expand_dims(x_offset, axis=0), [batch, 1, 1])
#     y_offset = tf.tile(tf.expand_dims(y_offset, axis=0), [batch, 1, 1])

#     x_icoords = x_bcoords + x_offset
#     y_icoords = y_bcoords + y_offset

#     # batch offset
#     batch_offset = tf.range(batch) * width * height
#     batch_offset = tf.tile(tf.expand_dims(batch_offset, axis=1), [1, rwidth*rheight])
#     batch_offset = tf.reshape(batch_offset, shape=(batch, rheight, rwidth))

#     return x_icoords, y_icoords, batch_offset, fgmask # [batch, rheight, rwidth]


# def heatmaps_to_reprojected_heatmaps(src_d_heatmaps, depth1s, c2Tc1s, intrinsics, block_size, name='HM2HM'):

#     '''
#     convert CNN output d_heatmaps to reprojected heatmaps

#     Args:
#         src_d_heatmaps: [batch, height/N, width/N, N*N+1], CNN output
#         depths1s: [batch, height, width, 1], depths of photos1
#         c2Tc1s: [batch, 4, 4], transformation matrix from camera1 to camera2
#         intrinsics: [batch, 3, 3], intrinsics matrix
#         block_size: N, downsampling rate (2**#pooling)
#     Return:
#         dst_heatmaps: [batch, height, width, 1]
#     '''
#     with tf.name_scope(name):
#         batch = tf.shape(depth1s)[0]
#         height = tf.shape(depth1s)[1]
#         width = tf.shape(depth1s)[2]

#         rheight = tf.shape(src_d_heatmaps)[1]
#         rwidth = tf.shape(src_d_heatmaps)[2]

#         # extract keypoints from each blocks
#         with tf.name_scope('KP-EXTRACT'):
#             x_src, y_src, b_offset, fgmask = extract_xy_coords(src_d_heatmaps, block_size)


#         with tf.name_scope('XY2IDX'):
#             # convert (x,y) to idx=x+y*width+b*width*height
#             x_src = tf.reshape(x_src, (batch, 1, -1))
#             y_src = tf.reshape(y_src, (batch, 1, -1))
#             b_offset = tf.reshape(b_offset, (batch, 1, -1))
#             idx_src = x_src + y_src * width + b_offset

#         with tf.name_scope('PIX2CAM'):
#             # extract depth on (x,y)
#             depth1s_flt = tf.reshape(depth1s, [-1])
#             z1_src = tf.gather(depth1s_flt, idx_src)

#             # pixel to camera coordinate (x,y,1 --> x,y,z,1)
#             ones = tf.ones_like(x_src)
#             pix_coords = tf.concat([x_src, y_src, ones], axis=1)
#             pix_coords = tf.cast(pix_coords, tf.float32)

#             inv_intrinsics = tf.matrix_inverse(intrinsics)
#             cam_coords = tf.matmul(inv_intrinsics, pix_coords) * z1_src
#             cam_coords = tf.concat([cam_coords, tf.cast(ones,tf.float32)], axis=1)


#         with tf.name_scope('TRANS-MAT'):
#             # get Transform matrix
#             filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
#             filler = tf.tile(filler, [batch, 1, 1])
#             intrinsics_4x4 = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
#             intrinsics_4x4 = tf.concat([intrinsics_4x4, filler], axis=1)
#             proj_mats = tf.matmul(intrinsics_4x4, c2Tc1s)

#         with tf.name_scope('CAM2PIX'):
#             # camera to pixel coordinate
#             unnormalized_pixel_coords = tf.matmul(proj_mats, cam_coords) # [batch, 4, N(=rwidth*rheight)]

#             x_u = tf.slice(unnormalized_pixel_coords, [0,0,0], [-1,1,-1])
#             y_u = tf.slice(unnormalized_pixel_coords, [0,1,0], [-1,1,-1])
#             z_u = tf.slice(unnormalized_pixel_coords, [0,2,0], [-1,1,-1])
#             epsilon = tf.constant(1e-10, dtype=tf.float32)
#             x_n = x_u / (z_u + epsilon)
#             y_n = y_u / (z_u + epsilon)

#         with tf.name_scope('SUBSAMPLING'):
#             # clipping coordinates
#             pad = 1
#             width2 = width + 2 * pad
#             height2 = height + 2 * pad
#             x_n = tf.clip_by_value(tf.cast(tf.round(x_n), tf.int32)+pad, 0, width+1) # 0~width+1 (actual range=1~width-1)
#             y_n = tf.clip_by_value(tf.cast(tf.round(y_n), tf.int32)+pad, 0, height+1)
#             zero = tf.constant(0, dtype=tf.int32)
#             zeros = tf.zeros_like(x_n)
#             fgmask = tf.reshape(fgmask, (batch, 1, -1))
#             x_n = tf.where(tf.equal(fgmask, zero), zeros, x_n)
#             y_n = tf.where(tf.equal(fgmask, zero), zeros, y_n)
#             x_n = tf.reshape(x_n, [batch, -1])
#             y_n = tf.reshape(y_n, [batch, -1])

#             b_offset = tf.range(batch) * width2 * height2
#             b_offset = tf.tile(tf.expand_dims(b_offset, axis=1), [1, rwidth*rheight])

#             idx_dst = x_n + y_n * width2 + b_offset
#             idx_dst = tf.reshape(idx_dst, [-1])

#             idx_dst, _ = tf.unique(idx_dst) # remove duplicate indices to apply sparce_to_dense

#             # ignore sorted order (https://stackoverflow.com/questions/35508894/sparse-to-dense-requires-indices-to-be-lexicographically-sorted-in-0-7)
#             dst_heatmaps = tf.sparse_to_dense(idx_dst, [batch*height2*width2], 1, 0, validate_indices=False)


#             dst_heatmaps = tf.reshape(dst_heatmaps, (batch, height2, width2))
#             dst_heatmaps = tf.slice(dst_heatmaps, [0,1,1], [-1,height,width])
#             dst_heatmaps = tf.cast(tf.expand_dims(dst_heatmaps, axis=-1), tf.float32)

#         return dst_heatmaps

def compute_multi_gradients(photos, pad, name='SMOOTHNESS'):
    with tf.name_scope(name):
        height = tf.shape(photos)[1]
        width = tf.shape(photos)[2]
        Dx = tf.zeros_like(photos)
        Dy = tf.zeros_like(photos)
        photos_pad = tf.pad(photos, [[0, 0], [pad, pad], [pad, pad], [0,0]], 'REFLECT')

        for i in range(pad):
            s = pad + i + 1
            Dx = Dx + tf.slice(photos_pad, [0,pad,s,0], [-1,height,width,-1])\
                    - tf.slice(photos_pad, [0,pad,i,0], [-1,height,width,-1])
            Dy = Dy + tf.slice(photos_pad, [0,s,pad,0], [-1,height,width,-1])\
                    - tf.slice(photos_pad, [0,i,pad,0], [-1,height,width,-1])
        Da = tf.sqrt(Dx**2+Dy**2)

    return Dx, Dy, Da

def compute_fg_mask_from_gradients(gradients, block_size, grad_thresh, reduce_op=tf.reduce_mean, name='FGMASK', keep_dims=False):

    with tf.name_scope(name):
        d_grads = tf.space_to_depth(gradients, block_size)
        d_grads = reduce_op(d_grads, axis=3, keep_dims=keep_dims)
        d_fgmask = tf.cast(tf.greater(d_grads, grad_thresh), tf.float32)

        # restore fgmask to original resolution
        # d_fgmask2 = tf.tile(tf.expand_dims(d_fgmask, -1), [1,1,1,block_size**2])
        # fgmask = tf.depth_to_space(d_fgmask2, block_size)

        return d_fgmask

def compute_background_loss(d_heatmaps, d_fgmask, name='BG-LOSS'):
    '''
    Args:
        d_heatmaps: [batch, height/N, width/N, N**2+1], heatmap logits
        d_fgmask: [batch, height/N,width/N,1], 0/1 mask
    '''
    with tf.name_scope(name):
        # fg_logits, bg_logits = tf.split(d_heatmaps, [-1,1], axis=-1)
        # d_bgmask = 1.0 - d_fgmask

        # I wonder sigmoid_cross_entropy is fine because d_heatmaps is a malti-class tensor originally
        # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_bgmask, logits=bg_logits) # shape is the same as inputs [batch,height/N, width/N, 1]
        # bg_count = tf.maximum(1.0, tf.reduce_sum(d_bgmask))
        # bg_loss = tf.div(tf.reduce_sum(d_bgmask * xentropy), bg_count) # not take whole mean

        # neumerically unstable way
        # https://www.tensorflow.org/get_started/mnist/beginners
        # fg_probs, bg_probs = tf.split(d_heatmaps, [-1,1], axis=-1)
        d_bgmask = 1.0 - d_fgmask

        # (1) tf sigmoid xentropy impl.
        # bg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_bgmask, logits=d_heatmaps))
        # (2) naive sigmoid xentropy impl.
        # bg_prob = tf.nn.sigmoid(d_heatmaps)
        # xentropy = -d_bgmask * tf.log(bg_prob) - (1-d_bgmask) * tf.log(1.0-bg_prob) # - y_label * log(y_pred)
        # bg_loss = tf.reduce_mean(xentropy)

        # fg_mask = 1.0 - tf.nn.sigmoid(d_heatmaps)

        # (3) tf softmax xentropy impl
        # import IPython
        # IPython.embed()
        d_prob = tf.nn.softmax(d_heatmaps) # normalize
        fg_prob, bg_prob = tf.split(d_prob, [-1,1], axis=-1)
        print('Map shape ', d_heatmaps.shape, d_prob.shape, bg_prob.shape)
        eps = 1e-6
        bg_prob = tf.clip_by_value(bg_prob, eps, 1.0-eps) # to avoid NaN at tf.log
        xentropy = -d_bgmask * tf.log(bg_prob) - (1-d_bgmask) * tf.log(1.0-bg_prob) # - y_label * log(y_pred)
        bg_loss = tf.reduce_mean(xentropy)
        fg_mask = 1.0 - bg_prob

        # bg_count = tf.maximum(1.0, tf.reduce_sum(d_bgmask))
        # bg_loss = tf.div(tf.reduce_sum(d_bgmask * xentropy), bg_count) # not take whole mean

        rheight = tf.shape(fg_mask)[1]
        rwidth = tf.shape(fg_mask)[2]
        block_size = 16
        fg_mask = tf.image.resize_images( fg_mask, (rheight*block_size, rwidth*block_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.summary.image('pred_fgmask2', fg_mask, max_outputs=4)

        return bg_loss

def compute_repeatable_loss(pred_d_heatmaps, trans_heatmaps, block_size, name='REPEAT-LOSS'):
    '''
    Args:
        pred_d_heatmaps: [batch, height/N, width/N, N**2+1]
        trans_heatmaps: [batch, height, width, 1]
    '''
    with tf.name_scope(name):
        trans_d_heatmaps = tf.space_to_depth(trans_heatmaps, block_size)
        kp_bg_map = tf.reduce_sum(trans_d_heatmaps, axis=-1, keep_dims=True)
        kp_bg_map = tf.cast(tf.less(kp_bg_map, 1.0), tf.float32)
        kp_fg_map = 1.0 - tf.squeeze(kp_bg_map, axis=-1)

        trans_d_heatmaps = tf.concat([trans_d_heatmaps, kp_bg_map], axis=3) # add BG channels

        xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=trans_d_heatmaps,
                                                        logits=pred_d_heatmaps) # shape = [batch,height/N,width/N]
        kp_count = tf.maximum(1.0, tf.reduce_sum(kp_fg_map))
        repeat_loss = tf.div(tf.reduce_sum(kp_fg_map * xentropy), kp_count)

        return repeat_loss

def get_R_loss(pred_d_heatmaps, trans_heatmaps, d_fgmask, weight_bg, block_size):

    # repeat_loss = compute_repeatable_loss(pred_d_heatmaps, trans_heatmaps, block_size, name='R-Loss')
    bg_loss = compute_background_loss(pred_d_heatmaps, d_fgmask, name='BG-Loss')

    # loss = repeat_loss + weight_bg * bg_loss
    loss = bg_loss

    # tf.summary.scalar('repeat_loss', repeat_loss)
    # tf.summary.scalar('bg_loss', bg_loss)
    tf.summary.scalar('loss', loss)

    return loss

def soft_max_and_argmax_1d(inputs, axis=-1, inputs_index=None, keep_dims=False, com_strength1=250.0, com_strength2=250.0):

    # Safe softmax
    inputs_exp1 = tf.exp(com_strength1*(inputs - tf.reduce_max(inputs, axis=axis, keep_dims=True)))
    inputs_softmax1 = inputs_exp1 / (tf.reduce_sum(inputs_exp1, axis=axis, keep_dims=True) + 1e-8)

    inputs_exp2 = tf.exp(com_strength2*(inputs - tf.reduce_max(inputs, axis=axis, keep_dims=True)))
    inputs_softmax2 = inputs_exp2 / (tf.reduce_sum(inputs_exp2, axis=axis, keep_dims=True) + 1e-8)

    inputs_max = tf.reduce_sum(inputs * inputs_softmax1, axis=axis, keep_dims=keep_dims)

    inputs_index_shp = [1,]*len(inputs.get_shape())
    inputs_index_shp[axis] = -1
    if inputs_index is None:
        inputs_index = tf.range(inputs.get_shape().as_list()[axis], dtype=inputs.dtype) # use 0,1,2,..,inputs.shape[axis]-1
    inputs_index = tf.reshape(inputs_index, inputs_index_shp)
    inputs_amax = tf.reduce_sum(inputs_index * inputs_softmax2, axis=axis, keep_dims=keep_dims)

    return inputs_max, inputs_amax

def soft_argmax_2d(patches_bhwc, patch_size, do_softmax=True, com_strength=10):
    # Returns the relative soft-argmax position, in the -1 to 1 coordinate
    # system of the patch

    width = patch_size
    height = patch_size

    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    xy_grid = tf.stack([x_t, y_t], axis=-1)[None] # BHW2

    maxes_bhwc = patches_bhwc
    if do_softmax:
        exps_bhwc = tf.exp(
                        com_strength*(patches_bhwc - tf.reduce_max(
                            patches_bhwc, axis=(1, 2), keep_dims=True)))
        maxes_bhwc = exps_bhwc / (
            tf.reduce_sum(exps_bhwc, axis=(1, 2), keep_dims=True) + 1e-8)

    dxdy = tf.reduce_sum(xy_grid * maxes_bhwc, axis=(1,2))

    return dxdy
