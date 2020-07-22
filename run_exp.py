from __future__ import division
from __future__ import print_function


import datetime
import argparse
import pprint

from misc.datasets import TextDataset
from model import CondGAN
from trainer import CondGANTrainer
from misc.get_configs import parse_args
from misc.utils import mkdir_p




if __name__ == "__main__":
    args = parse_args()
    print(args)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    dataset = TextDataset(datadir='datasets/'+args.dataset+'/')

    print("Dataset created!")
    dataset.train = dataset.get_data()

    model = CondGAN(args, image_shape=dataset.image_shape)
    print("model created!")

    # if args.for_training:
    ckt_logs_dir = "ckt_logs/%s" % \
        ("{}_logs".format(args.dataset))
    res_dir = "retrieved_res/%s" % \
        ("{}_res".format(args.dataset))
    mkdir_p(ckt_logs_dir)
    mkdir_p(res_dir)
    with open(ckt_logs_dir + '/args.txt', 'w') as fid:
        fid.write(str(args)+'\n')


    algo = CondGANTrainer(
        args,
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir,
        res_dir=res_dir
    )
    print("Trainer initialized!")
   
    algo.train()
