import os
import sys
try:
    import yaml
except:
    os.system('pip install yaml')
import argparse
import tensorflow as tf
from termcolor import colored

from hed.models.vgg16 import Vgg16
from hed.utils.io import IO
from hed.data.data_parser import DataParser
from time import strftime, localtime


class HEDTrainer():

    def __init__(self,dataDir=None,saveDir=None,initmodelfile=None,configfile=None):

        self.io = IO()
        self.init = True
        self.dataDir = dataDir
        self.saveDir = saveDir
        self.initmodelfile = initmodelfile
        try:
            pfile = open(configfile)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            print('Error reading config file {}, {}'.format(configfile, err))

    def setup(self):

        try:

            self.model = Vgg16(self.cfgs,self.saveDir,self.initmodelfile)
            self.io.print_info('Done initializing VGG-16 model')
            dirs = ['train', 'val', 'test', 'models']
            save_dir = self.saveDir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            dirs = [os.path.join(save_dir + '/{}'.format(d)) for d in dirs]
            _ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]

        except Exception as err:

            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self, session):

        if not self.init:
            return

        train_data = DataParser(self.cfgs,self.dataDir)

        self.model.setup_training(session)

        opt = tf.train.AdamOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        train = opt.minimize(self.model.loss)

        session.run(tf.global_variables_initializer())

        for idx in range(self.cfgs['max_iterations']):

            im, em, _ = train_data.get_training_batch()

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, summary, loss,fuse = session.run([train, self.model.merged_summary, self.model.loss, self.model.fuse],
                                           feed_dict={self.model.images: im, self.model.edgemaps: em},
                                           options=run_options,
                                           run_metadata=run_metadata)
            self.model.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(idx))
            time = strftime("%Y-%m-%d %H:%M:%S", localtime())
            info = '[{0}][INFO] {1}'.format(time, '[{}/{}] TRAINING loss : {}'.format(idx, self.cfgs['max_iterations'], loss))
            print(colored(info, 'green'))
            print(colored(opt._lr, 'green'))

            if idx % 100 == 0:
                self.model.train_writer.add_summary(summary, idx)
            '''
            if idx % self.cfgs['save_interval'] == 0:

                saver = tf.train.Saver()
                save_dir = self.saveDir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                saver.save(session, os.path.join(save_dir, 'models/detect-model'), global_step=idx)
            
            if idx % self.cfgs['val_interval'] == 0:

                im, em, _ = train_data.get_validation_batch()

                summary, error = session.run([self.model.merged_summary, self.model.error], feed_dict={self.model.images: im, self.model.edgemaps: em})
                if idx % 1000 == 0:
                    self.model.val_writer.add_summary(summary, idx)
                self.io.print_info('[{}/{}] VALIDATION error : {}'.format(idx, self.cfgs['max_iterations'], error))
            '''
            if idx == self.cfgs['max_iterations'] - 1:
                save_dir = self.saveDir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                graph = tf.graph_util.convert_variables_to_constants(session, session.graph_def, ["fuse"])
                tf.train.write_graph(graph, os.path.join(save_dir, 'models'), 'testgraph.pb', as_text=False)

        self.model.train_writer.close()
