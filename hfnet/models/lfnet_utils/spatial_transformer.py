import tensorflow as tf


def _meshgrid(height, width):
    with tf.name_scope('meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

def transformer_crop(images, out_size, batch_inds, kpts_xy, kpts_scale=None, kpts_ori=None, thetas=None, name='SpatialTransformCropper'):
    # images : [B,H,W,C]
    # out_size : (out_width, out_height)
    # batch_inds : [B*K,] tf.int32 [0,B)
    # kpts_xy : [B*K,2] tf.float32 or whatever
    # kpts_scale : [B*K,] tf.float32
    # kpts_ori : [B*K,2] tf.float32 (cos,sin)
    if isinstance(out_size, int):
        out_width = out_height = out_size
    else:
        out_width, out_height = out_size
    hoW = out_width // 2
    hoH = out_height // 2

    with tf.name_scope(name):

        num_batch = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        C = tf.shape(images)[3]
        num_kp = tf.shape(batch_inds)[0] # B*K
        zero = tf.zeros([], dtype=tf.int32)
        max_y = tf.cast(tf.shape(images)[1] - 1, tf.int32)
        max_x = tf.cast(tf.shape(images)[2] - 1, tf.int32)

        grid = _meshgrid(out_height, out_width) # normalized -1~1
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.stack([num_kp]))
        grid = tf.reshape(grid, tf.stack([num_kp, 3, -1]))

        # create 6D affine from scale and orientation
        # [s, 0, 0]   [cos, -sin, 0]
        # [0, s, 0] * [sin,  cos, 0]
        # [0, 0, 1]   [0,    0,   1]
        #
        if thetas is None:
            thetas = tf.eye(2,3, dtype=tf.float32)
            thetas = tf.tile(thetas[None], [num_kp,1,1])
            if kpts_scale is not None:
                thetas = thetas * kpts_scale[:,None,None]
            ones = tf.tile(tf.constant([[[0,0,1]]], tf.float32), [num_kp,1,1])
            thetas = tf.concat([thetas, ones], axis=1) # [num_kp, 3,3]

            if kpts_ori is not None:
                cos = tf.slice(kpts_ori, [0,0], [-1,1]) # [num_kp, 1]
                sin = tf.slice(kpts_ori, [0,1], [-1,1])
                zeros = tf.zeros_like(cos)
                ones = tf.ones_like(cos)
                R = tf.concat([cos,-sin,zeros, sin,cos,zeros, zeros,zeros,ones], axis=-1)
                R = tf.reshape(R, [-1,3,3])
                thetas = tf.matmul(thetas, R)
        # Apply transformation to regular grid
        T_g = tf.matmul(thetas, grid) # [num_kp,3,3] * [num_kp,3,H*W]
        x = tf.slice(T_g, [0,0,0], [-1,1,-1]) # [num_kp,1,H*W]
        y = tf.slice(T_g, [0,1,0], [-1,1,-1])

        # unnormalization [-1,1] --> [-out_size/2,out_size/2]
        x = x * out_width / 2.0
        y = y * out_height / 2.0

        if kpts_xy.dtype != tf.float32:
            kpts_xy = tf.cast(kpts_xy, tf.float32)

        kp_x_ofst = tf.expand_dims(tf.slice(kpts_xy, [0,0], [-1,1]), axis=1) # [B*K,1,1]
        kp_y_ofst = tf.expand_dims(tf.slice(kpts_xy, [0,1], [-1,1]), axis=1) # [B*K,1,1]

        # centerize on keypoints
        x = x + kp_x_ofst
        y = y + kp_y_ofst
        x = tf.reshape(x, [-1]) # num_kp*out_height*out_width
        y = tf.reshape(y, [-1])

        # interpolation
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width*height
        base = tf.tile(batch_inds[:,None], [1, out_height*out_width]) # [B*K,out_height*out_width]
        base = tf.reshape(base, [-1]) * dim1
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = tf.reshape(images, tf.stack([-1, C])) # [B*height*width,C]
        im_flat = tf.cast(im_flat, tf.float32)

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        x0_f = tf.cast(x0, tf.float32)
        x1_f = tf.cast(x1, tf.float32)
        y0_f = tf.cast(y0, tf.float32)
        y1_f = tf.cast(y1, tf.float32)

        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        output = tf.reshape(output, tf.stack([num_kp, out_height, out_width, C]))
        output.set_shape([batch_inds.shape[0], out_height, out_width, images.shape[-1]])
        return output

# def inplane_inverse_warp(images, thetas=None, name='InplaneInverseWarp'):
#     # Args
#     #   images : [B,H,W,C], tf.float32
#     #   thetas : [B,3,3], tf.float32
#     # Return
#     #   outputs: [B,H,W,C], tf.float32

#     # need visibility masks

#     with tf.name_scope(name):
#         batch_size = tf.shape(images)[0]
#         height ,width = images.get_shape().as_list()[1:3] # scalar
#         cy = height / 2
#         cx = width / 2
#         batch_inds = tf.range(batch_size)
#         kpts_xy = tf.cast(tf.tile(tf.stack([cx, cy])[None], [batch_size, 1]), tf.float32)
#         return transformer_crop(images, (width, height), batch_inds, kpts_xy, thetas=thetas)

def inplane_inverse_warp(images, thetas, out_size=None, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    images : [B,H,W,C] float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    thetas: [B,3,3] float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (width, height)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``thetas`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        thetas = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_width = out_size[0]
            out_height = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            inside_x = tf.logical_and(tf.greater_equal(x0, 0), tf.less(x1, width))
            inside_y = tf.logical_and(tf.greater_equal(y0, 0), tf.less(y1, height))
            visibility = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32) # [N]
            visibility = tf.reshape(visibility, [num_batch, out_height, out_width, 1])
            visibility.set_shape([images.get_shape()[0], out_height, out_width, 1])

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1


            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output, visibility

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(thetas, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            # thetas = tf.reshape(thetas, (-1, 2, 3))
            # thetas = tf.cast(thetas, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_width = out_size[0]
            out_height = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(thetas, grid) # [B,3,3], # [B,3,1]
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s / (z_s+1e-6), [-1])
            y_s_flat = tf.reshape(y_s / (z_s+1e-6), [-1])

            input_transformed, visibility = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))

            return output, visibility

    with tf.variable_scope(name):
        if out_size is None:
            out_size = images.get_shape().as_list()[1:3][::-1] # width,height

        output, visibility = _transform(thetas, images, out_size)
        out_width, out_height = out_size
        output.set_shape([images.get_shape()[0], out_height, out_width, images.get_shape()[-1]])
        visibility.set_shape([images.get_shape()[0], out_height, out_width, 1])

        return output, visibility


# def inplane_inverse_warp(images, out_size, thetas, name='InplaneInverseWarp'):
#     # Args
#     #   images : [B,H,W,C], tf.float32
#     #   thetas : [B,3,3], tf.float32
#     # Return
#     #   outputs: [B,H,W,C], tf.float32

#     # need visibility masks

#     # with tf.name_scope(name):
#     #     batch_size = tf.shape(images)[0]
#     #     height ,width = images.get_shape().as_list()[1:3] # scalar
#     #     cy = height / 2
#     #     cx = width / 2
#     #     batch_inds = tf.range(batch_size)
#     #     kpts_xy = tf.cast(tf.tile(tf.stack([cx, cy])[None], [batch_size, 1]), tf.float32)
#     #     return transformer_crop(images, (width, height), batch_inds, kpts_xy, thetas=thetas)

#     if isinstance(out_size, int):
#         out_width = out_height = out_size
#     else:
#         out_width, out_height = out_size

#     with tf.name_scope(name):

#         batch_size = tf.shape(images)[0]
#         height = tf.shape(images)[1]
#         width = tf.shape(images)[2]
#         height_f = tf.cast(height, tf.float32)
#         width_f = tf.cast(width, tf.float32)
#         C = tf.shape(images)[3]
#         batch_inds = tf.range(batch_size)
#         zero = tf.zeros([], dtype=tf.int32)
#         max_y = tf.cast(tf.shape(images)[1] - 1, tf.int32)
#         max_x = tf.cast(tf.shape(images)[2] - 1, tf.int32)

#         grid = _meshgrid(out_height, out_width) # normalized -1~1
#         grid = tf.expand_dims(grid, 0)
#         grid = tf.reshape(grid, [-1])
#         grid = tf.tile(grid, tf.stack([batch_size]))
#         grid = tf.reshape(grid, tf.stack([batch_size, 3, -1]))

#         # Apply transformation to regular grid
#         T_g = tf.matmul(thetas, grid) # [B,3,3] * [B,3,H*W]
#         x = tf.slice(T_g, [0,0,0], [-1,1,-1]) # [B,1,H*W]
#         y = tf.slice(T_g, [0,1,0], [-1,1,-1])

#         # unnormalization [-1,1] --> [0, out_size]
#         x = (x+1.0) * out_width / 2.0
#         y = (y+1.0) * out_height / 2.0
#         x = tf.reshape(x, [-1]) # B*out_height*out_width
#         y = tf.reshape(y, [-1])

#         # interpolation
#         x0 = tf.cast(tf.floor(x), tf.int32)
#         x1 = x0 + 1
#         y0 = tf.cast(tf.floor(y), tf.int32)
#         y1 = y0 + 1

#         x0 = tf.clip_by_value(x0, zero, max_x)
#         x1 = tf.clip_by_value(x1, zero, max_x)
#         y0 = tf.clip_by_value(y0, zero, max_y)
#         y1 = tf.clip_by_value(y1, zero, max_y)

#         inside_x = tf.logical_and(tf.greater_equal(x0, 0), tf.less(x1, out_width))
#         inside_y = tf.logical_and(tf.greater_equal(y0, 0), tf.less(y1, out_height))
#         visibility = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32) # [N]
#         visibility = tf.reshape(visibility, [batch_size, out_height, out_width, 1])
#         visibility.set_shape([images.get_shape()[0], out_height, out_width, 1])

#         dim2 = width
#         dim1 = width*height
#         base = tf.tile(batch_inds[:,None], [1, out_height*out_width]) # [B*K,out_height*out_width]
#         base = tf.reshape(base, [-1]) * dim1
#         base_y0 = base + y0 * dim2
#         base_y1 = base + y1 * dim2
#         idx_a = base_y0 + x0
#         idx_b = base_y1 + x0
#         idx_c = base_y0 + x1
#         idx_d = base_y1 + x1

#         im_flat = tf.reshape(images, tf.stack([-1, C])) # [B*height*width,C]
#         im_flat = tf.cast(im_flat, tf.float32)

#         Ia = tf.gather(im_flat, idx_a)
#         Ib = tf.gather(im_flat, idx_b)
#         Ic = tf.gather(im_flat, idx_c)
#         Id = tf.gather(im_flat, idx_d)

#         x0_f = tf.cast(x0, tf.float32)
#         x1_f = tf.cast(x1, tf.float32)
#         y0_f = tf.cast(y0, tf.float32)
#         y1_f = tf.cast(y1, tf.float32)

#         wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
#         wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
#         wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
#         wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)

#         output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
#         output = tf.reshape(output, tf.stack([batch_size, out_height, out_width, C]))
#         output.set_shape([images.get_shape()[0], out_height, out_width, images.get_shape()[-1]])
#         return output, visibility

def inplane_coordinate_warp(kpts1, batch_inds, thetas, img_width, img_height, name='InplaneCoordWarp'):
    # Args
    #   kpts1: [N,2] int32 or whatever (x,y)
    #   batch_inds: [N,] int32 [0,batch_size)
    #   intrinsics_3x3: [B,3,3] float32
    #   thetas: [N,3,3] tf.float32
    # Return
    #   kpts: [N,2] int32 (x,y)
    #   visibility: [N,] float32

    with tf.name_scope(name):
        N = tf.shape(kpts1)[0]
        ones = tf.ones([N,1], dtype=tf.float32)
        kpts1 = tf.cast(kpts1, tf.float32)
        kpts_x = tf.slice(kpts1, [0,0], [-1,1])
        kpts_y = tf.slice(kpts1, [0,1], [-1,1])
        kpts_x = (2 * kpts_x / img_width) - 1.0 # [0,width) --> [-1,1)
        kpts_y = (2 * kpts_y / img_height) - 1.0 # [0,width) --> [-1,1)
        norm_kpts = tf.expand_dims(tf.concat([kpts_x, kpts_y, ones], axis=1), axis=-1) # [N,3] --> [N,3,1]
        # kpts1 = tf.expand_dims(tf.concat([kpts1, ones], axis=1), axis=-1) # [N,3]->[N,3,1]

        # inv_intrinsics = tf.matrix_inverse(intrinsics_3x3) # [B,3,3]
        # gathered_intrinsics = tf.gather(intrinsics_3x3, batch_inds) # [B,4,4],[N] --> [N,4,4]
        # gatherd_inv_intrinsics = tf.gather(inv_intrinsics, batch_inds) # [B,3,3],[N]-->[N,3,3]

        # T = tf.matmul(gathered_intrinsics, tf.matmul(thetas, gatherd_inv_intrinsics)) # [N,3,3]
        # trans_kpts = tf.matmul(T, kpts1)[...,0] # [N,3,1] --> [N,3]
        trans_kpts = tf.matmul(thetas, norm_kpts)[...,0]

        x_u = tf.slice(trans_kpts, [0,0], [-1,1]) # [N,1]
        y_u = tf.slice(trans_kpts, [0,1], [-1,1])
        z_u = tf.slice(trans_kpts, [0,2], [-1,1])

        x_n = x_u / (z_u+1e-6)
        y_n = y_u / (z_u+1e-6)

        x_n = (x_n+1.0) * img_width / 2.0 # [-1,1]-->[0,width]
        y_n = (y_n+1.0) * img_height / 2.0

        # check visibility
        x0 = tf.cast(tf.floor(x_n), tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y_n), tf.int32)
        y1 = y0 + 1

        inside_x = tf.logical_and(tf.greater_equal(x0, 0), tf.less(x1, img_width))
        inside_y = tf.logical_and(tf.greater_equal(y0, 0), tf.less(y1, img_height))
        visibility = tf.cast(tf.logical_and(inside_x, inside_y), tf.float32) # [N,1]
        visibility = tf.squeeze(visibility, 1) # [N] float32

        trans_kpts_safe = tf.concat([
                                tf.clip_by_value(tf.cast(tf.round(x_n), tf.int32), 0, img_width-1),
                                tf.clip_by_value(tf.cast(tf.round(y_n), tf.int32), 0, img_height-1),
                            ], axis=1) # [N,2]
        return trans_kpts_safe, visibility


'''
## TEST

# data loader
img_dir = '/cvlabdata2/home/ono/Datasets/VOT2017/bolt1'
in_images = []
for i in range(1,300,50):
    img = imread(os.path.join(img_dir, '{:08d}.jpg'.format(i)))
    in_images.append(img)
in_images = np.array(in_images).astype(np.float32)

crop_size = (64, 64)
out_width, out_height = crop_size
hoW = out_width // 2
hoH = out_height // 2

in_batch, in_height, in_width = in_images.shape[:3]
in_kp_coords = []
in_kp_scales = []
in_kp_oris = []
in_kp_inds = []

fix_theta = np.pi / 2

mX, mY = np.meshgrid(np.arange(in_width), np.arange(in_height))

canvas = in_images.copy()

in_patches = []
in_x = []
in_y = []
for n in range(100):
    b = np.random.choice(in_batch)
    kpy = np.random.randint(out_height, in_height-out_height)
    kpx = np.random.randint(out_width, in_width-out_width)
    kps = np.random.random() + 0.5
    kp_theta = 2*(np.random.random()-0.5)*np.pi/2.0 # [-pi/2,pi/2]
    in_kp_coords.append([kpx, kpy])
    in_kp_scales.append(kps)
#     in_kp_oris = [math.cos(kp_theta), math.sin(kp_theta)]
    in_kp_oris.append([math.cos(fix_theta), math.sin(fix_theta)])
    in_kp_inds.append(b)

    hw = int(hoW * kps)
    hh = int(hoH * kps)
    pt1 = (kpx-hw, kpy-hh)
    pt2 = (kpx+hw, kpy+hh)
    cv2.rectangle(canvas[b], pt1, pt2, (0,0xFF,0))
    patch = in_images[b, pt1[1]:pt2[1], pt1[0]:pt2[0],:]
    patch = cv2.resize(patch, (out_height, out_width))
    in_patches.append(patch)
    in_x.append(mX[pt1[1]:pt2[1], pt1[0]:pt2[0]])
    in_y.append(mY[pt1[1]:pt2[1], pt1[0]:pt2[0]])
in_kp_coords = np.array(in_kp_coords)
in_kp_scales = np.array(in_kp_scales)
in_kp_inds = np.array(in_kp_inds)
in_patches = np.array(in_patches)
in_x = np.array(in_x)[...,None]
in_y = np.array(in_y)[...,None]

tf.reset_default_graph()
images = tf.placeholder(tf.float32, [None, None, None, 3]) # [B,H,W,C]
batch_inds = tf.placeholder(tf.int32, [None]) # [B*K]
kpts_xy = tf.placeholder(tf.float32, [None, 2]) # [B*K, 2] x,y unnormalized coords
kpts_scale = tf.placeholder(tf.float32, [None]) # [B*K]
kpts_ori = tf.placeholder(tf.float32, [None,2]) #[B*K,2]
num_batch = tf.shape(images)[0]
num_kp = tf.shape(kpts_xy)[0]

trans = transformer_crop(images, crop_size, batch_inds, kpts_xy, kpts_scale, kpts_ori)

with tf.Session() as sess:
    feed_dict = {
        images: in_images,
        batch_inds: in_kp_inds,
        kpts_xy: in_kp_coords.astype(np.float32),
        kpts_scale: in_kp_scales,
        kpts_ori: in_kp_oris
    }
    outs = sess.run(trans, feed_dict)

canvas = np.concatenate([outs, in_patches], axis=2)
display_image_batch(canvas)
'''
