from __future__ import division
from __future__ import print_function


import numpy as np
import cPickle as pkl
import random


class Dataset(object):
    def __init__(self, image_dict, imsize, embedding_dict, embed_size, datadir='CUB/'):
        self._image_dict = image_dict
        self._embedding_dict = embedding_dict
        self._embed_size = embed_size
        self._epochs_completed = -1
        self._num_examples = 100
        self.num_classes = len(self._embedding_dict.keys())

        with open(datadir+'trainids.pkl', 'rb') as fid:
            self._trainids = pkl.load(fid)
        with open(datadir+'testids.pkl', 'rb') as fid:
            self._testids = pkl.load(fid)


        self.max_txt_emb_per_cls = {}
        self.curr_txt_emb_per_cls = {}
        for cls, emb_list in self.embedding_dict.iteritems():
            self.max_txt_emb_per_cls[cls] = len(emb_list)
            self.curr_txt_emb_per_cls[cls] = 0

        self.max_img_emb_per_cls = {}
        self.curr_img_emb_per_cls = {}
        for cls, emb_list in self.image_dict.iteritems():
            self.max_img_emb_per_cls[cls] = len(emb_list)
            self.curr_img_emb_per_cls[cls] = 0

        self._index_in_epoch = self._num_examples
        self._imsize = imsize
        self._perm = None
        self.cls_index = 0

    @property
    def image_dict(self):
        return self._image_dict

    @property
    def embedding_dict(self):
        return self._embedding_dict

    @property
    def imsize(self):
        return self._imsize

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def testids(self):
        return self._testids

    @property
    def trainids(self):
        return self._trainids


    def next_batch(self, batch_size=32):
        """Return the next `batch_size` examples from this data set."""

        num_classes = len(self._trainids)

        if batch_size >= num_classes:

            num_batch_cls = int(batch_size / num_classes)
            num_rem_batch_cls = int(batch_size % num_classes)
            perm_cls = random.sample(self._trainids, num_rem_batch_cls)
            tot_cls_list = self._trainids * num_batch_cls + perm_cls
            self._perm = np.random.permutation(tot_cls_list)

        else:
            
            if self.cls_index >= num_classes:
                
                self.cls_index = 0
                random.shuffle(self._trainids)
 
            self._perm = self._trainids[self.cls_index: self.cls_index + batch_size]
            self.cls_index +=  batch_size


        images = np.zeros((batch_size, self._imsize))
        embeddings = np.zeros((batch_size, self._embed_size))
        labels = np.zeros((batch_size), dtype=int)
        similar_images = np.zeros((batch_size, self._imsize))
        for i, idx in enumerate(self._perm):
            self.curr_img_emb_per_cls[idx] = (self.curr_img_emb_per_cls[idx] + 1) % self.max_img_emb_per_cls[idx]
            ix_view = self.curr_img_emb_per_cls[idx]
            images[i,:] = self._image_dict[idx][ix_view]
            ix_view += 1
            ix_view = ix_view % self.max_img_emb_per_cls[idx]
            similar_images[i, :] = self._image_dict[idx][ix_view]

            embeddings[i,:] = self._embedding_dict[idx][0]
            labels[i] = idx
 
        return images, np.roll(embeddings, 1, 0), np.roll(images, 1, 0), embeddings, labels



class TextDataset(object):
    def __init__(self, datadir='CUB/'):
        self.image_shape = None
        self.image_dim =  None
        self.embedding_shape = None
        self.train = None
        self.test = None
        self.datadir = datadir
        self.embedding_filename = datadir+'txt_embed_dict.pkl'
        self.image_embedding_file = datadir+'image_embed_dict.pkl'
        self.get_data()

    def get_data(self):
        with open(self.embedding_filename, 'rb') as fid:
            embedding_dict = pkl.load(fid)
        with open(self.image_embedding_file, 'rb') as fid:
            image_dict = pkl.load(fid)

        self.image_shape = image_dict[image_dict.keys()[0]][0].shape[0] 
        self.image_dim = self.image_shape
 
        self.embedding_shape = embedding_dict[embedding_dict.keys()[0]][0].shape[0]
        return Dataset(image_dict, self.image_shape, embedding_dict, self.embedding_shape, datadir=self.datadir)

