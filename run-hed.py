# LIBRARY_PATH=/usr/local/cuda/lib64
import os
import sys
import argparse
import tensorflow as tf
from hed.utils.io import IO
from hed.test import HEDTester
from hed.train import HEDTrainer

def get_session(gpu_fraction):

    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = 4
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def main(args):

    if not (args.run_train or args.run_test):
        print('Set AtLeast one of the options --train | --test ')
        parser.print_help()
        return

    if args.run_test or args.run_train:
        session = get_session(args.gpu_limit)

    if args.run_train:
        trainer = HEDTrainer(args.datadir,args.savedir,args.initmodelfile,args.configfile)
        trainer.setup()
        trainer.run(session)

    if args.run_test:
        tester = HEDTester(args.datadir,args.savedir,args.configfile)
        tester.setup(session)
        tester.run(session)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility for Training/Testing DL models(Concepts/Captions) using theano/keras')
    parser.add_argument('--configfile', dest='configfile', type=str, help='Experiment configuration file')
    parser.add_argument('--train', dest='run_train', action='store_true', default=False, help='Launch training')
    parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='Launch testing on a list of images')
    parser.add_argument('--gpu-limit', dest='gpu_limit', type=float, default=1.0, help='Use fraction of GPU memory (Useful with TensorFlow backend)')

    parser.add_argument('--datadir', dest='datadir', type=str, help='Experiment configuration datadir path')
    parser.add_argument('--savedir', dest='savedir', type=str, help='Experiment configuration savedir path')
    parser.add_argument('--initmodelfile', dest='initmodelfile', type=str, help='Experiment init model file')

    args = parser.parse_args()

    main(args)
