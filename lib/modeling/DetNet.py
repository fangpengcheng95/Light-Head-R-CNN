# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Implements ResNet and ResNeXt.

See: https://arxiv.org/abs/1512.03385, https://arxiv.org/abs/1611.05431.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def add_ResNet18_conv4_body(model):
    return add_ResNet18_convX_body(model, (2, 2, 2))


def add_ResNet18_conv5_body(model):
    return add_ResNet18_convX_body(model, (2, 2, 2, 2))



def add_ResNet50_conv4_body(model):
    #return add_ResNet_convX_body(model, (3, 4, 6))
    return add_ResNet18_convX_body(model, (3, 4, 6))


def add_ResNet50_conv5_body(model):
    return add_ResNet18_convX_body(model, (3, 4, 6, 3, 3))


def add_ResNet101_conv4_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23))


def add_ResNet101_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23, 3))


def add_ResNet152_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    # e.g., prefix = res2
    for i in range(n):
	stride_init=stride_init if i == 0 else 1
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            #'{}_unit{}'.format(prefix, i+1),
            blob_in,
            dim_in,
            dim_out,
            dim_inner,
            dilation,
            stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_out
    return blob_in, dim_in


def add_ResNet18_convX_body(model, block_counts, freeze_at=2):
    """Add a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    assert freeze_at in [0, 2, 3, 4, 5]
    p = model.Conv('data', 'conv1', 3, 32, 3, pad=1, stride=2, no_bias=1)
    p = model.AffineChannel(p, 'bn1', dim=32, inplace=True)
    p = model.Relu(p, p)
    p = model.Conv('conv1', 'conv2', 32, 32, 3, pad=1, stride=1, no_bias=1)
    p = model.AffineChannel(p, 'bn2', dim=32, inplace=True)
    p = model.Relu(p, p)
    p = model.Conv('conv2', 'conv3', 32, 64, 3, pad=1, stride=1, no_bias=1)
    p = model.AffineChannel(p, 'bn3', dim=64, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
       
    dim_in = 64
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'layer1', p, n1, dim_in, 256, dim_bottleneck, 1, stride_init=1)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'layer2', s, n2, dim_in, 512, dim_bottleneck * 2, 1, stride_init=2
    )
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'layer3', s, n3, dim_in, 1024, dim_bottleneck * 4, 1, stride_init=2
    )
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 5:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'layer4', s, n4, dim_in, 1024, dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION, stride_init=1
        )
        if freeze_at == 5:
            model.StopGradient(s, s)
        n5 = block_counts[4]
        s, dim_in = add_stage(
            model, 'layer5', s, n5, dim_in, 1024, dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION, stride_init=1
        )
        return s, dim_in, 1. / 32. * cfg.RESNETS.RES5_DILATION
    else:
        return s, dim_in, 1. / 16.


def add_ResNet_convX_body(model, block_counts, freeze_at=2):
    """Add a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    assert freeze_at in [0, 2, 3, 4, 5]
    p = model.Conv('data', 'conv1', 3, 64, 7, pad=3, stride=2, no_bias=1)
    p = model.AffineChannel(p, 'res_conv1_bn', dim=64, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    dim_in = 64
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'res2', p, n1, dim_in, 256, dim_bottleneck, 1)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res3', s, n2, dim_in, 512, dim_bottleneck * 2, 1
    )
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res4', s, n3, dim_in, 1024, dim_bottleneck * 4, 1
    )
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'res5', s, n4, dim_in, 2048, dim_bottleneck * 8,
            cfg.RESNETS.RES5_DILATION
        )
        if freeze_at == 5:
            model.StopGradient(s, s)
        return s, dim_in, 1. / 32. * cfg.RESNETS.RES5_DILATION
    else:
        return s, dim_in, 1. / 16.


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        #model, 'res5', 'pool5', 3, dim_in, 2048, dim_bottleneck * 8, 1,
        model, 'stage4', 'pool5', 3, dim_in, 2048, dim_bottleneck * 8, 1,
        stride_init
    )
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048
    #return s, 512


def add_ResNet_roi_conv5_light_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        #model, 'res5', 'pool5', 3, dim_in, 2048, dim_bottleneck * 8, 2,
        model, 'stage4', blob_in, 3, dim_in, 2048, dim_bottleneck * 8, 2,
        stride_init
    )
    dim_mid = 64
    x1_1 = model.Conv(
        s,
        'convx1_1',
        dim_in,
        dim_mid,
        kernel=[15, 1],
        pad_t = 7,
        pad_b = 7,
        pad_r = 0,
        pad_l = 0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    x1_2 = model.Conv(
        x1_1,
        'convx1_2',
        dim_mid,
        490,
        kernel=[1, 15],
        pad_t = 0,
        pad_b = 0,
        pad_r = 7,
        pad_l = 7,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    x2_1 = model.Conv(
        s,
        'convx2_1',
        dim_in,
        dim_mid,
        kernel=[1, 15],
        pad_t = 0,
        pad_b = 0,
        pad_r = 7,
        pad_l = 7,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    x2_2 = model.Conv(
        x2_1,
        'convx2_2',
        dim_mid,
        490,
        kernel=[15, 1],
        pad_t = 7,
        pad_b = 7,
        pad_r = 0,
        pad_l = 0,
        stride=1,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    light_head = model.net.Sum(['convx1_2', 'convx2_2'], 'light_head') 
    
    if cfg.FAST_RCNN.ROI_XFORM_METHOD == 'PSRoIPool':
        model.net.PSRoIPool(
            [light_head, 'rois'], ['pool5', '_mapping_channel'],
            group_size=7,
            output_dim=10,
            spatial_scale=spatial_scale
        )
        roi_dim = 10
    else:
        roi_feat = model.RoIFeatureTransform(
            light_head,
            'pool5',
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        roi_dim = 490

    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    fc7 = model.FC('pool5', 'fc7', roi_dim * roi_size * roi_size, 2048)
    fc7 = model.Relu('fc7', 'fc7')
    return fc7, 2048



def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2,
    inplace_sum=False
):
    """Add a residual block to the model."""
    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # Max pooling is performed prior to the first stage (which is uniquely
    # distinguished by dim_in = 64), thus we keep stride = 1 for the first stage
    #stride = stride_init if (
    #    dim_in != dim_out and dim_in != 64 and dilation == 1
    #) else 1
    stride = stride_init

    # transformation blob
    tr = globals()[cfg.RESNETS.TRANS_FUNC](
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        group=cfg.RESNETS.NUM_GROUPS,
        dilation=dilation
    )

    # sum -> ReLU
    sc = add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride, dilation)
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        #s = model.net.Sum([tr, sc], prefix + '_sum')
        s = model.net.Sum([tr, sc], prefix + '_add')

    return model.Relu(s, s)


def add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride, dilation):
    if dim_in == dim_out and dilation == 1:
        return blob_in
    if prefix[-1] != '0':
        return blob_in

    c = model.Conv(
        blob_in,
        #prefix + '_branch1',
        #prefix + '_conv1sc',
	prefix + '_downsample_0', 
        dim_in,
        dim_out,
        kernel=1,
        stride=stride,
        no_bias=1
    )
    return model.AffineChannel(c, prefix + '_downsample_1', dim=dim_out)


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def bottleneck_v3_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1,
):
    """Add a bottleneck transformation to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
    dim_inner = int(dim_out*0.25)
    if dilation == 2:
        str3x3 = 1
    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffine_v2(
        blob_in,
        prefix,
        dim_in,
        dim_inner,
        kernel=1,
        stride=1,
        pad=0,
	suffix='1',
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine_v2(
        cur,
        prefix,
        dim_inner,
        dim_inner,
        kernel=3,
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
	suffix='2',
        inplace=True
    )
    cur = model.Relu(cur, cur)
    cur = model.ConvAffine_v2(
        cur,
        prefix,
        dim_inner,
        dim_out,
        kernel=1,
        stride=1,
        pad=0,
	suffix='3',
        inplace=False
    )

    return cur


def bottleneck_v2_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1,
):
    """Add a bottleneck transformation to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
    dim_inner = dim_out
    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffine_v2(
        blob_in,
        prefix,
        dim_in,
        dim_inner,
        kernel=3,
        stride=stride,
        pad=1,
	suffix='1',
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine_v2(
        cur,
        prefix,
        dim_inner,
        dim_out,
        kernel=3,
        stride=1,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
	suffix='2',
        inplace=False
    )
    #cur = model.Relu(cur, cur)

    return cur

def bottleneck_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffine(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        stride=str1x1,
        pad=0,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
        inplace=True
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> BN (no ReLU)
    # NB: for now this AffineChannel op cannot be in-place due to a bug in C2
    # gradient computation for graphs like this
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        stride=1,
        pad=0,
        inplace=False
    )
    return cur
