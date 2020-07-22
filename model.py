#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.

"""
file containing the main proposed model class
Major chunks of code taken from https://github.com/hanzhanggit/StackGAN-Pytorch/blob/master/code/model.py
"""

# import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range


from misc.kl_div import KL_loss
class CondGAN(object):
    def __init__(self, args, image_shape, num_classes=200):
        self.batch_size = args.batch_size
        self.image_shape = image_shape
        self.gf_dim = args.gf_dim
        self.df_dim = args.df_dim
        self.ef_dim = args.embed_dim
        self.num_classes = num_classes


    def generate_condition_and_KL_loss(self, c_var, reuse=False, isTrainable=True):
        with tf.variable_scope("g_net", reuse=reuse):
            conditions = tf.layers.dense(inputs=c_var, units=self.ef_dim*2, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                  activation=tf.nn.leaky_relu, name='kl_div_fc1',trainable=isTrainable,reuse=reuse)

            mean = conditions[:, :self.ef_dim]
            log_sigma = conditions[:, self.ef_dim:]
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(log_sigma)
            c = mean + stddev * epsilon
            m = (mean + log_sigma) / 2.
            kl_div_loss = 0.5 * (KL_loss(mean, m) + KL_loss(log_sigma, m))
            return c, kl_div_loss

    def embed_Image(self, img_embed, reuse=False, isTrainable=True):
        with tf.variable_scope("CSEM_net", reuse=reuse):
            net = tf.layers.dense(inputs=img_embed, units=self.ef_dim, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                  activation=tf.nn.leaky_relu, name='embed_fc',trainable=isTrainable,reuse=reuse)

            return net



    def generator(self, z_var, reuse=False,isTrainable=True):
        with tf.variable_scope("g_net", reuse=reuse):
            net = tf.layers.dense(inputs=z_var, units=self.gf_dim, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                  activation=tf.nn.leaky_relu, name='g_fc1',trainable=isTrainable,reuse=reuse)

            net = tf.layers.dense(inputs=net, units=self.gf_dim*2, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                                  activation=tf.nn.leaky_relu, name='g_fc3',trainable=isTrainable,reuse=reuse)

            net = tf.layers.dense(inputs = net, units = self.image_shape, \
                                  kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02), \
                                  activation=tf.nn.relu,name='g_fc2', trainable = isTrainable, reuse=reuse)

            return net


    def nnLayer(self, inp, reuse=False, isTrainable=True):
        net = tf.layers.dense(inputs=inp, units=self.image_shape, \
                              kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                              activation=tf.nn.relu, name='fc1',trainable=isTrainable,reuse=reuse)
        return net

 

    def _discriminator(self, x_c_code, reuse, isTrainable):
        net = tf.layers.dense(inputs=x_c_code, units=self.df_dim, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=tf.nn.leaky_relu, name='d_fc1',trainable=isTrainable,reuse=reuse)

        logit = tf.layers.dense(inputs=net, units=1, \
                            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.02),  \
                            activation=None, name='d_rf',trainable=isTrainable,reuse=reuse)
        return logit
    def get_discriminator(self, image_var, embed_var, reuse=False,isTrainable=True):
        with tf.variable_scope("d_net", reuse=reuse):
            x_c_code = tf.concat([image_var, embed_var], 1)
            return self._discriminator(x_c_code, reuse, isTrainable)


