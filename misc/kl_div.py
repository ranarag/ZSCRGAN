#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.

"""
script to calculate the KL divergence
"""
import tensorflow as tf

def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + 0.5 * (-1 + tf.exp(2. * log_sigma) + \
                                    tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


