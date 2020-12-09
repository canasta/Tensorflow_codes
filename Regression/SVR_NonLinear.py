from os import makedirs
import os.path
import logging

import numpy as np
import tensorflow as tf

class TFSVRAnalyzer(object):
    """ SVR 모델을 관리하는 클래스 """
    def __init__(
        self,
        X_shape,
        Y_shape,
        sw_name=None,
        save_path='./save/model/',
        cv=5,
        batch_size=50,
        epsilon=0.5):

        self.sess = tf.Session()
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        
        self.X_data = tf.placeholder(shape=[None, X_shape[-1]], dtype=tf.float32)
        self.Y_target = tf.placeholder(shape=[None, Y_shape[-1]], dtype=tf.float32)

        # RBF kernel
        gamma = tf.constant(-50.)
        dist = tf.reduce_sum(tf.square(self.X_data), 1)
        dist = tf.reshape(dist, [-1,1])
        sq_dist = tf.add(tf.subtract(dist, tf.multifly(2., tf.matmul(self.X_data, tf.transpose(self.X_data)))), tf.transpose(dist))
        self.kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dist)))
        #TODO SOMETHING
        
        self._W = tf.Variable(tf.random_normal(shape=[X_shape[1], Y_shape[1]]), name='W')
        self._b = tf.Variable(tf.random_normal(shape=[X_shape[0], Y_shape[1]]), name='b')

        self.output = tf.add(tf.matmul(self.X_data, self._W), self._b)
        self.epsilon = tf.constant([epsilon], name='epsilon')

        self.loss = tf.reduce_mean(
            tf.maximum(
                0.,
                tf.subtract(tf.abs(tf.subtract(self.output, self.Y_target)), self.epsilon)
            )
        )

        self.sw_name = sw_name
        self.save_path = save_path
        self.is_fit = False
        self.batch_size = batch_size

    def fit(self, x, y, epoch=1e6):

        self.opt = tf.train.GradientDescentOptimizer(0.075)
        train_step = self.opt.minimize(self.loss)

        for i in range(epoch):
            rand_idx = np.random.choice(len(x), size=self.batch_size)
            rand_x = np.transpose([x[rand_idx]])
            rand_y = np.transpose([y[rand_idx]])
            self.sess.run(
                train_step,
                feed_dict={self.X_data: rand_x, self.Y_target: rand_y}
            )

            if (i+1)%1e4 == 0:
                _loss = self.sess.run(
                    self.loss,
                    feed_dict={
                        self.X_data: np.transpose([x]),
                        self.Y_target: np.transpose([y])
                    }
                )

                print(f'{i:>{int(np.floor(np.log10(epoch)))+1}}/{epoch} Train Loss={str(_loss)}')

    def predict(self, x):
        [[slope]] = self.sess.run(self._W)
        [[y_intercept]] = self.sess.run(self._b)
        width = self.sess.run(self.epsilon)

        res = []
        res_max = []
        res_min = []

        for i in x:
            res.append(slope*i+y_intercept)
            res_max.append(slope*i+y_intercept+width)
            res_min.append(slope*i+y_intercept-width)

        return res#, res_max, res_min
    
    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        makedirs(save_path, exist_ok=True)
        try:
            saver = tf.train.Saver()
            saver.save(self.sess, os.path.join(save_path, f'{self.sw_name}_svr'))
        except Exception as ex:
            logger = logging.getLogger('sequential_analysis')
            logger.error('error on save_model: {}'.format(ex))
        
    def load_model(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        path = os.path.join(save_path, f'{self.sw_name}_svr.meta')
        if os.path.exists(path):
            try:
                saver = tf.train.import_meta_graph(f'{self.sw_name}_svr.meta')
                saver.restore(self.sess, tf.train.latest_checkpoint(save_path))

                graph = tf.get_default_graph()
                self._W = graph.get_tensor_by_name('W1:0')
                self._b = graph.get_tensor_by_name('b1:0')
                self.epsilon = graph.get_tensor_by_name('epsilon1:0')
                self.is_fit = True
            except Exception as ex:
                logger = logging.getLogger('sequential_analysis')
                logger.error('error on load_model: {}'.format(ex))
        return self.is_fit
