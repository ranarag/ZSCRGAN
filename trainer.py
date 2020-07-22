#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.

"""
file containing code to train the GAN model
"""





import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
import cPickle as pkl
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
class CondGANTrainer(object):
    def __init__(self, args, model,
                 dataset=None, exp_name="model2",
                 ckt_logs_dir="ckt_logs",
                 res_dir="res"):
        self.model = model
        self.dataset = dataset
        print(self.dataset.embedding_shape)
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.res_dir = res_dir
        self.args = args
        self.batch_size = self.args.batch_size
        self.max_epoch = self.args.epochs
        self.log_vars = []
        self.flat_image_embed_list = []
        for class_label, embedding_list in self.dataset.train.image_dict.iteritems():
            if class_label not in self.dataset.train.testids:
                continue
            for embed in embedding_list:
                self.flat_image_embed_list.append((class_label, embed))
        self.test_img_X = np.zeros((len(self.flat_image_embed_list), self.dataset.image_shape))
        self.test_img_Y = np.zeros((len(self.flat_image_embed_list)), dtype=int)

        for i, (label, embed) in enumerate(self.flat_image_embed_list):
            self.test_img_X[i,:] = embed
            self.test_img_Y[i]= label
        unique, counts = np.unique(self.test_img_Y, return_counts=True)
        self.img_per_class = dict(zip(unique, counts))
        flat_image_embed_list = []
        for class_label, embedding_list in self.dataset.train.image_dict.iteritems():
            if class_label not in self.dataset.train.trainids:
                continue
            for embed in embedding_list:
                flat_image_embed_list.append((class_label, embed))
        self.X = np.zeros((len(flat_image_embed_list), self.dataset.image_shape))
        self.Y = np.zeros((len(flat_image_embed_list)), dtype=int)

        for i, (label, embed) in enumerate(flat_image_embed_list):
            self.X[i,:] = embed
            self.Y[i]= label

        flat_txt_embed_list = []
        for class_label, embedding_list in self.dataset.train.embedding_dict.iteritems():
            if class_label not in self.dataset.train.testids:
                continue
            for embed in embedding_list:
                flat_txt_embed_list.append((class_label, embed))

        self.embed_X = np.zeros((len(flat_txt_embed_list), self.dataset.embedding_shape)) 
        self.label_Y = np.zeros((len(flat_txt_embed_list)), dtype=int)

        for i, (label, embed) in enumerate(flat_txt_embed_list):
            self.embed_X[i,:] = embed
            self.label_Y[i] = label



    def build_placeholder(self):
        self.real_images = tf.placeholder(tf.float32, [None,\
                                          self.dataset.image_shape],
                                          name='real_images')

        self.wrong_images = tf.placeholder(tf.float32, [None,\
                                           self.dataset.image_shape],
                                           name='wrong_images')
        self.embeddings = tf.placeholder(tf.float32, [None,\
                                                      self.dataset.embedding_shape],
                                         name='input_embeddings')
        self.wrong_embeddings = tf.placeholder(tf.float32, [None,\
                                                      self.dataset.embedding_shape],
                                         name='wrong_embeddings')
        self.labels = tf.placeholder(tf.int32, [None])

        self.generator_lr = tf.placeholder(tf.float32, [],
                                           name='generator_learning_rate')
        self.discriminator_lr = tf.placeholder(tf.float32, [],
                                               name='discriminator_learning_rate')
        self.CSEM_lr = tf.placeholder(tf.float32, [],
                                      name='CSEM_learning_rate')
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z')
        self.z_1 = tf.placeholder(tf.float32, [None, self.args.z_dim], name='noise_z_1')

        self.acc = tf.placeholder(tf.float32, [], name="Accuracy")


    def init_op(self):
        self.build_placeholder()
        self.c1, kl_loss = self.model.generate_condition_and_KL_loss(self.embeddings)
        wrong_c1, wrong_kl_loss = self.model.generate_condition_and_KL_loss(self.wrong_embeddings, reuse=True)
        c2, _ = self.model.generate_condition_and_KL_loss(self.embeddings, reuse=True, isTrainable=False)
        self.log_vars.append(("hist_z", self.z))

        fake_imgs = self.model.generator(tf.concat([self.c1, self.z], 1))
        wrong_fake_imgs = self.model.generator(tf.concat([wrong_c1, self.z_1], 1), reuse=True, isTrainable=True)


        discriminator_loss, generator_loss, \
            csem_loss = self.compute_losses(fake_imgs, wrong_fake_imgs)
        generator_loss += (self.args.kl_div_coefficient * (kl_loss + wrong_kl_loss) / 2.0)
 
        self.log_vars.append(("g_loss_kl_loss", kl_loss))
        self.log_vars.append(("g_loss", generator_loss))
        self.log_vars.append(("d_loss", discriminator_loss))
  
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_net')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_net')
        self.CSEM_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CSEM_net')

        self.prepare_trainer(generator_loss, discriminator_loss, csem_loss)
        self.define_summaries()

        self.fake_imgs = self.model.generator(tf.concat([c2, self.z], 1), reuse=True, isTrainable=False)

        ######################### for  testing ###############################
        self.gen_embeds = self.model.embed_Image(self.fake_imgs, reuse=True, isTrainable=False)
        self.resize_imgs = self.model.embed_Image(self.real_images, reuse=True, isTrainable=False)
        ######################################################################

        


    def compute_losses(self, fake_images, wrong_fake_images):


        realConstructionLoss = tf.reduce_mean(tf.reduce_sum(tf.abs(fake_images - self.real_images), axis=1))

        wrongConstructionLoss = tf.reduce_mean(tf.reduce_sum(tf.abs(fake_images - self.wrong_images), axis=1))

        maxMarginRegularizer = realConstructionLoss - wrongConstructionLoss + 2.0

        real_logit = self.model.get_discriminator(self.real_images, self.embeddings)
        fake_logit = self.model.get_discriminator(fake_images, self.embeddings, reuse=True)
 
        wrong_logit = self.model.get_discriminator(self.wrong_images, self.embeddings, reuse=True)


        discriminator_loss = (tf.reduce_mean(fake_logit) + tf.reduce_mean(wrong_logit))/2. - tf.reduce_mean(real_logit)

        normalize_a = tf.nn.l2_normalize(self.model.embed_Image(fake_images),1)
        normalize_b = tf.nn.l2_normalize(self.c1,1)
        normalize_c = tf.nn.l2_normalize(self.model.embed_Image(wrong_fake_images, reuse=True), 1)
        realcosineSimilarity = tf.reduce_mean(tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=1))

        wrongcosineSimilarity = tf.reduce_mean(tf.reduce_sum(tf.multiply(normalize_c,normalize_b), axis=1))
 
        csem_loss = tf.math.log(1.0 + tf.math.exp(wrongcosineSimilarity - realcosineSimilarity))

        generator_loss = -tf.reduce_mean(fake_logit) + csem_loss 

        generator_loss += self.args.mm_reg_coeff * maxMarginRegularizer


        return discriminator_loss, generator_loss, csem_loss




    def prepare_trainer(self, generator_loss, discriminator_loss, csem_loss):
        generator_opt = tf.train.RMSPropOptimizer(self.generator_lr)


        generator_grad_vars = generator_opt.compute_gradients(generator_loss,
                                                    		 var_list=self.g_vars)

        self.generator_trainer = generator_opt.apply_gradients(generator_grad_vars)

        CSEM_opt = tf.train.AdamOptimizer(self.CSEM_lr, beta1=0.5)

        CSEM_grad_vars = CSEM_opt.compute_gradients(csem_loss,
                                                    		 var_list=self.CSEM_vars)

        self.CSEM_trainer = CSEM_opt.apply_gradients(CSEM_grad_vars)

        self.discriminator_opt = tf.train.RMSPropOptimizer(self.discriminator_lr).minimize(discriminator_loss,
                                                                                           var_list=self.d_vars)
        self.discriminator_clipper = [var.assign(tf.clip_by_value(var, -self.args.clip_val, self.args.clip_val))
                                      for var in self.d_vars]



    def define_summaries(self):
        all_sum = {'g':[], 'd': [], 'hist':[]}

        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            else:
                all_sum['hist'].append(tf.summary.histogram(k, v))


        self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.acc_sum = tf.summary.scalar("Accuracy_sum", self.acc)
        self.hist_sum = tf.summary.merge(all_sum['hist'])


    def build_model(self, sess, model_path=''):
        self.init_op()
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        sess.run(tf.global_variables_initializer(), options=run_opts)
        if len(model_path) > 0:
            print "Reading model parameters from {}".format(model_path)
            restore_vars = tf.global_variables()
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, model_path)

            istart = model_path.rfind('_') + 1
            iend = model_path.rfind('.')
            counter = int(model_path[istart:iend])
        else:
            print "Created model with fresh parameters."
            counter = 0
        return counter

 
    def train(self):
        self.accuracy = 0.0
        snapshot_path = ''
        generator_lr = self.args.generator_lr
        CSEM_lr = self.args.CSEM_lr
        discriminator_lr = self.args.discriminator_lr
        number_example = 531000
        epoch = 0
        keys = ['d_loss', 'g_loss']
        log_vars = []
        log_keys = []
        for k, v in self.log_vars:
            if k in keys:
                log_vars.append(v)
                log_keys.append(k)
        updates_per_epoch = int(number_example / self.batch_size)


        # creating session
        acc_list = []
        acc_counter = 0
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            counter = self.build_model(sess, snapshot_path)
            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_net')
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)



    	    for j in range(1, 10000):
                epoch += 1
                all_log_vals = []
                
                ################## E-Step #########################
                for i in range(updates_per_epoch):
                    for _ in range(100):
                        #### discriminator training step ##########
                        for disc_iter in range(5):
                            images, wrong_embeds, wrong_images, embeddings,labels =\
                                self.dataset.train.next_batch(self.batch_size)
                            z_rand = np.random.normal(0, 1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                            z_rand_1 = np.random.normal(0,1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                            feed_dict = {self.real_images: images,
                                         self.wrong_images: wrong_images,
                                         self.embeddings: embeddings,
                                         self.wrong_embeddings: wrong_embeds,
                                         self.z: z_rand,
                                         self.z_1: z_rand_1,                                      
                                         self.labels: labels,
                                         self.discriminator_lr: discriminator_lr
                                        }
                            feed_out = [self.discriminator_opt,
                                        self.d_sum,
                                        self.hist_sum,
                                        log_vars]
                            
                            _, d_sum, hist_sum, log_vals = sess.run(feed_out,
                                                                    feed_dict)
                            sess.run(self.discriminator_clipper)
                            summary_writer.add_summary(d_sum, counter)
                            summary_writer.add_summary(hist_sum, counter)
                            all_log_vals.append(log_vals)
                            
                            
                    ############ generator training step#######################
                        images, wrong_embeds, wrong_images, embeddings,labels =\
                            self.dataset.train.next_batch(self.batch_size)
                        z_rand = np.random.normal(0, 1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                        z_rand_1 = np.random.normal(0, 1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                        feed_dict = {self.real_images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.wrong_embeddings: wrong_embeds,
                                     self.z: z_rand,
                                     self.z_1: z_rand_1,
                                     self.labels: labels,
                                     self.generator_lr: generator_lr,
                                    }
                        feed_out = [self.generator_trainer,
                                    self.g_sum]
                        _, g_sum = sess.run(feed_out,
                                            feed_dict)

                        summary_writer.add_summary(g_sum, counter)

                    
                    #################### M-Step ######################################
                    for _ in range(100):
                        images, wrong_embeds, wrong_images, embeddings,labels =\
                            self.dataset.train.next_batch(self.batch_size)
                        z_rand = np.random.normal(0, 1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                        z_rand_1 = np.random.normal(0, 1., [self.batch_size, self.args.z_dim]).astype(np.float32)
                        feed_dict = {self.real_images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.wrong_embeddings: wrong_embeds,
                                     self.z: z_rand,
                                     self.z_1: z_rand_1,
                                     self.labels: labels,
                                     self.CSEM_lr: CSEM_lr
                                     }
                        feed_out = self.CSEM_trainer
                        _ = sess.run(feed_out,
                                            feed_dict)

   
 
                    counter += 1

                #################### Evaluation Part###########
                    
                    acc_counter += 1
                    new_accuracy = self.evaluate_model(sess)
                    acc_list.append(new_accuracy)
                                                        
                    acc_summary = sess.run(self.acc_sum, {self.acc: new_accuracy})
                    summary_writer.add_summary(acc_summary, acc_counter)
                    if new_accuracy > self.accuracy:

                        print "New best Prec@50  = {}".format(new_accuracy)
                        self.accuracy = new_accuracy
                        snapshot_path = "%s/acc_%s_%s_%s.ckpt" %(self.log_dir,
                                                            new_accuracy,
                                                            self.exp_name,
                                                            str(counter))

                        avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                        dic_logs = {}
                        for k, v in zip(log_keys, avg_log_vals):
                            dic_logs[k] = v

                            log_line = ";".join("%s: %s" %(str(k), str(dic_logs[k]))
                                                            for k in dic_logs)
                            print "Epoch {} | {}".format(epoch, log_line)



    def evaluate_model(self, sess):

        z_rand = np.random.normal(0, 1., [self.embed_X.shape[0], self.args.z_dim]).astype(np.float32)
        new_embeddings, new_test_img_X= sess.run([self.gen_embeds, self.resize_imgs], feed_dict={self.embeddings:self.embed_X,
                                                                               self.real_images: self.test_img_X,
                                                             self.z: z_rand})

        cosine_sim = cosine_similarity(new_embeddings, new_test_img_X)
        success_dict =  {}
        retrieved_dict = {}
        for  idx in range(new_embeddings.shape[0]):
            idx = self.label_Y[idx]
            success_dict [idx] = 0
            retrieved_dict[idx] = 0

        ret_list = []

        for i in range(new_embeddings.shape[0]):
            txt_class_label = self.label_Y[i]
            sim_list = []

            for j in range(self.test_img_X.shape[0]):
                sim = cosine_sim[i,j]
                im_class_label = self.test_img_Y[j]
                sim_list.append((im_class_label, sim))

            sim_list = sorted(sim_list, key= lambda x: x[1], reverse=True)

            im_list = []
            retrieve_count = 50

            for im_label, _ in sim_list[:retrieve_count]:

                im_list.append(im_label)
                retrieved_dict[txt_class_label] += 1
                if txt_class_label == im_label:
                    success_dict[txt_class_label] += 1

            ret_list.append({txt_class_label: im_list})
 
        avg_p = 0.0
        for idx in retrieved_dict.keys():

            avg_p = avg_p + (success_dict[idx] / float(retrieved_dict[idx]))

        avg_p = avg_p / len(retrieved_dict)
        if self.accuracy < avg_p:
            out_file = open("{0}/acc{1}.pkl".format(self.res_dir, avg_p), "wb")

            pkl.dump(ret_list, out_file)

            out_file.close()
        return avg_p

 
