# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

from hed.losses import sigmoid_cross_entropy_balanced
from hed.utils.io import IO


class Vgg16():

    def __init__(self, cfgs,saveDir=None,initmodelfile=None, run='training'):

        self.cfgs = cfgs
        self.saveDir = saveDir
        self.io = IO()
        weights_file = initmodelfile
        self.data_dict = np.load(weights_file, encoding='latin1').item()
        self.io.print_info("Model weights loaded from {}".format(self.cfgs['model_weights_path']))

        self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']],name="inputlayer")
        self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])

        self.define_model()

    def define_model(self):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """

        start_time = time.time()

        self.conv1_1 = self.conv_layer_vgg(self.images, "conv1_1")
        self.conv1_2 = self.conv_layer_vgg(self.conv1_1, "conv1_2")
        self.side_1 = self.side_layer(self.conv1_2, "side_1", 1)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

        self.conv2_1 = self.conv_layer_vgg(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer_vgg(self.conv2_1, "conv2_2")
        self.side_2 = self.side_layer(self.conv2_2, "side_2", 2)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

        self.conv3_1 = self.conv_layer_vgg(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer_vgg(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer_vgg(self.conv3_2, "conv3_3")
        self.side_3 = self.side_layer(self.conv3_3, "side_3", 4)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

        self.conv4_1 = self.conv_layer_vgg(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer_vgg(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer_vgg(self.conv4_2, "conv4_3")
        self.side_4 = self.side_layer(self.conv4_3, "side_4", 8)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

        self.conv5_1 = self.conv_layer_vgg(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer_vgg(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer_vgg(self.conv5_2, "conv5_3")
        self.side_5 = self.side_layer(self.conv5_3, "side_5", 16)

        self.io.print_info('Added CONV-BLOCK-5+SIDE-5')

        self.side_outputs = [self.side_1, self.side_2, self.side_3, self.side_4, self.side_5]

        w_shape = [1, 1, len(self.side_outputs), 1]
        self.fuse = self.conv_layer(tf.concat(self.side_outputs, axis=3),
                                    w_shape, name='fuse_1', use_bias=False,
                                    w_init=tf.constant_initializer(0.2))

        self.io.print_info('Added FUSE layer')

        # complete output maps from side layer and fuse layers
        self.outputs = self.side_outputs + [self.fuse]

        self.data_dict = None
        self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))

    


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer_vgg(self, bottom, name):
        """
            Adding a conv layer + weight parameters from a dict
        """
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def conv_layer(self, x, W_shape, b_shape=None, name=None,
                   padding='SAME', use_bias=True, w_init=None, b_init=None):

        W = self.weight_variable(W_shape, w_init)
        tf.summary.histogram('weights_{}'.format(name), W)
        if use_bias:
            b = self.bias_variable([b_shape], b_init)
            tf.summary.histogram('biases_{}'.format(name), b)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
        return conv + b if use_bias else conv

    def deconv_layer(self, x, upscale, name, padding='SAME', w_init=None):

        x_shape = tf.shape(x)
        in_shape = x.shape.as_list()
        w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
        strides = [1, upscale, upscale, 1]
        W = self.weight_variable(w_shape, w_init)
        tf.summary.histogram('weights_{}'.format(name), W)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
        deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)
        return deconv

    def side_layer(self, inputs, name, upscale):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
        """
        with tf.variable_scope(name):
            in_shape = inputs.shape.as_list()
            w_shape = [1, 1, in_shape[-1], 1]
            classifier = self.conv_layer(inputs, w_shape, b_shape=1,
                                         w_init=tf.constant_initializer(),
                                         b_init=tf.constant_initializer(),
                                         name=name + '_reduction')

            classifier = self.deconv_layer(classifier, upscale=upscale,
                                           name='{}_deconv_{}'.format(name, upscale),
                                           w_init=tf.truncated_normal_initializer(stddev=0.1))
            return classifier

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


    def weight_variable(self, shape, initial):
        init = initial(shape)
        return tf.Variable(init)

    def bias_variable(self, shape, initial):
        init = initial(shape)
        return tf.Variable(init)

    def setup_testing(self, session):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """
        self.predictions = []
        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """

        self.predictions = []
        self.loss = 0
        self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))
        for idx, b in enumerate(self.side_outputs):
            tf.summary.image('output_{}'.format(idx), b)
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        #tf.summary.image('fuse', self.fuse)
        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        tf.summary.image('fuseSigmoid', fuse_output)
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')
        #tf.summary.image('cvSource', self.images)
        tf.summary.image('cvSourceEdge', self.edgemaps)
        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)

        pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)

        self.merged_summary = tf.summary.merge_all()

        save_dir = self.saveDir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.train_writer = tf.summary.FileWriter(save_dir + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(save_dir + '/val')

    '''
    for bn test 
    '''
    def build(self,input_image,is_training):
        with tf.name_scope('processing'):
            #bgr cv2
            b,g,r=tf.split(input_image,3,axis=3)
            image=tf.concat([
                    b*0.00390625,
                    g*0.00390625,
                    r*0.00390625],axis=3)
        # vgg16
        # block 1
        self.conv1_1=self.conv_bn_f(image,is_training=is_training,name='conv1_1')
        self.conv1_2=self.conv_bn_f(self.conv1_1,is_training=is_training,name='conv1_2')
        self.pool1=self.max_pool(self.conv1_2,name='pool1')
        # block 2
        self.conv2_1=self.conv_bn_f(self.pool1,is_training=is_training,name='conv2_1')
        self.conv2_2=self.conv_bn_f(self.conv2_1,is_training=is_training,name='conv2_2')
        self.pool2=self.max_pool(self.conv2_2,name='pool2')
        # block 3
        self.conv3_1=self.conv_bn_f(self.pool2,is_training=is_training,name='conv3_1')
        self.conv3_2=self.conv_bn_f(self.conv3_1,is_training=is_training,name='conv3_2')
        self.conv3_3=self.conv_bn_f(self.conv3_2,is_training=is_training,name='conv3_3')
        self.pool3=self.max_pool(self.conv3_3,name='pool3')
        # block 4
        self.conv4_1=self.conv_bn_f(self.pool3,is_training=is_training,name='conv4_1')
        self.conv4_2=self.conv_bn_f(self.conv4_1,is_training=is_training,name='conv4_2')
        self.conv4_3=self.conv_bn_f(self.conv4_2,is_training=is_training,name='conv4_3')
        self.pool4=self.max_pool(self.conv4_3,name='pool4')
        # block 5
        self.conv5_1=self.conv_bn_f(self.pool4,is_training=is_training,name='conv5_1')
        self.conv5_2=self.conv_bn_f(self.conv5_1,is_training=is_training,name='conv5_2')
        self.conv5_3=self.conv_bn_f(self.conv5_2,is_training=is_training,name='conv5_3')

        self.upscore_dsn1_1=self.conv_bn(self.conv1_1,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_1')
        self.upscore_dsn1_2=self.conv_bn(self.conv1_2,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_2')
        
        self.score_dsn2_1=self.conv_bn(self.conv2_1,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_1')
        self.upscore_dsn2_1=self.upsampling(self.score_dsn2_1,tf.shape(image)[1:3])
        
        self.score_dsn2_2=self.conv_bn(self.conv2_2,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_2')
        self.upscore_dsn2_2=self.upsampling(self.score_dsn2_2,tf.shape(image)[1:3])
        
        self.score_dsn3_1=self.conv_bn(self.conv3_1,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_1')
        self.upscore_dsn3_1=self.upsampling(self.score_dsn3_1,tf.shape(image)[1:3])
        
        self.score_dsn3_2=self.conv_bn(self.conv3_2,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_2')
        self.upscore_dsn3_2=self.upsampling(self.score_dsn3_2,tf.shape(image)[1:3])
        
        self.score_dsn3_3=self.conv_bn(self.conv3_3,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_3')
        self.upscore_dsn3_3=self.upsampling(self.score_dsn3_3,tf.shape(image)[1:3])
        
        self.score_dsn4_1=self.conv_bn(self.conv4_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_1')
        self.upscore_dsn4_1=self.upsampling(self.score_dsn4_1,tf.shape(image)[1:3])
        
        self.score_dsn4_2=self.conv_bn(self.conv4_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_2')
        self.upscore_dsn4_2=self.upsampling(self.score_dsn4_2,tf.shape(image)[1:3])
        
        self.score_dsn4_3=self.conv_bn(self.conv4_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_3')
        self.upscore_dsn4_3=self.upsampling(self.score_dsn4_3,tf.shape(image)[1:3])
                
        self.score_dsn5_1=self.conv_bn(self.conv5_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_1')
        self.upscore_dsn5_1=self.upsampling(self.score_dsn5_1,tf.shape(image)[1:3])
        
        self.score_dsn5_2=self.conv_bn(self.conv5_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_2')
        self.upscore_dsn5_2=self.upsampling(self.score_dsn5_2,tf.shape(image)[1:3])
        
        self.score_dsn5_3=self.conv_bn(self.conv5_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_3')
        self.upscore_dsn5_3=self.upsampling(self.score_dsn5_3,tf.shape(image)[1:3])
        
        self.concat=tf.concat([self.upscore_dsn1_1,self.upscore_dsn1_2,self.upscore_dsn2_1,self.upscore_dsn2_2,self.upscore_dsn3_1,self.upscore_dsn3_2,self.upscore_dsn3_3,
                                   self.upscore_dsn4_1,self.upscore_dsn4_2,self.upscore_dsn4_3,self.upscore_dsn5_1,self.upscore_dsn5_2,self.upscore_dsn5_3],axis=3)
        
        self.score=self.conv_bn(self.concat,ksize=[1,1,13,self.class_number],is_training=is_training,name='score')
        self.softmax=tf.nn.softmax(self.score+tf.constant(1e-4))
        
        self.pred=tf.argmax(self.softmax,axis=-1)


    '''
    tf.contrib.layers.batch_norm(inputs, decay=0.999, center=True, scale=False, epsilon=0.001, activation_fn=None,
    param_initializers=None, param_regularizers=None, updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=True,
    reuse=None, variables_collections=None, outputs_collections=None, trainable=True, batch_weights=None, fused=None,
    data_format=DATA_FORMAT_NHWC, zero_debias_moving_mean=False, scope=None, renorm=False, renorm_clipping=None, renorm_decay=0.99, adjustment=None)

    tf.contrib.layers.batch_norm 参数：
    1 inputs： 输入
    2 decay ：衰减系数。合适的衰减系数值接近1.0,特别是含多个9的值：0.999(默认值),0.99,0.9。如果训练集表现很好而验证/测试集表现得不好，选择小的系数（推荐使用0.9）。如果想要提高稳定性，zero_debias_moving_mean设为True
    3 center：如果为True，有beta偏移量；如果为False，无beta偏移量
    4 scale：如果为True，则乘以gamma。如果为False，gamma则不使用。当下一层是线性的时（例如nn.relu），由于缩放可以由下一层完成，所以可以禁用该层。
    5 epsilon：避免被零除
    6 activation_fn：用于激活，默认为线性激活函数
    7 param_initializers ： beta, gamma, moving mean and moving variance的优化初始化
    8 param_regularizers ： beta and gamma正则化优化
    9 updates_collections ：Collections来收集计算的更新操作。updates_ops需要使用train_op来执行。如果为None，则会添加控件依赖项以确保更新已计算到位。
    10 is_training:图层是否处于训练模式。在训练模式下，它将积累转入的统计量moving_mean并 moving_variance使用给定的指数移动平均值 decay。当它不是在训练模式，那么它将使用的数值moving_mean和moving_variance。
    11 scope：可选范围variable_scope
    注意：训练时，需要更新moving_mean和moving_variance。默认情况下，更新操作被放入tf.GraphKeys.UPDATE_OPS，所以需要添加它们作为依赖项train_op。例如：  
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  with tf.control_dependencies(update_ops):    train_op = optimizer.minimize(loss)
    可以将updates_collections = None设置为强制更新，但可能会导致速度损失，尤其是在分布式设置中。
    '''    
    def conv_bn_f(self,bottom,is_training,name):
        # finu-tune and batch_norm ; fine-tune not shape,shape had known
        with tf.variable_scope(name):
            weights=self.get_conv_filter_v2(name)
            out=tf.nn.conv2d(bottom,filter=weights,strides=[1,1,1,1],padding='SAME')
            biases=self.get_bias_v2(name)
            out=tf.nn.bias_add(out,biases)
            #bn before relu and train True test False
            out=tf.contrib.layers.batch_norm(out,center=True,scale=True,is_training=is_training) # 这里的scale为真需要注意下，神马意思
            out=tf.nn.relu(out)
        return out
    
    def conv_bn(self,bottom,ksize,is_training,name):
        # initialize and batch_norm ; stride =[1,1,1,1]
        with tf.variable_scope(name):
            weights=tf.get_variable('weights',ksize,tf.float32,initializer=xavier_initializer())
            biases=tf.get_variable('biases',[ksize[-1]],tf.float32,initializer=tf.constant_initializer(0.0))
            out=tf.nn.conv2d(bottom,filter=weights,strides=[1,1,1,1],padding='SAME')
            out=tf.nn.bias_add(out,biases)
            #bn
            out=tf.contrib.layers.batch_norm(out,center=True,scale=True,is_training=is_training)
            out=tf.nn.relu(out)
        return out

    def get_conv_filter_v2(self,name):
        init=tf.constant_initializer(self.vgg16_params[name]['weights'])
        shape=self.vgg16_params[name]['weights'].shape
        var=tf.get_variable('weights',shape=shape,dtype=tf.float32,initializer=init)
        return var
    

    def get_bias_v2(self,name):
        init=tf.constant_initializer(self.vgg16_params[name]['biases'])
        shape=self.vgg16_params[name]['biases'].shape # tuple
        bias=tf.get_variable('biases',shape=shape,dtype=tf.float32,initializer=init)
        return bias

    def upsampling(self,bottom,feature_map_size):
        # feature_map_size: int [h,w]
        return tf.image.resize_bilinear(bottom,size=feature_map_size)

