import random
import sys
import os
import time

import numpy as np
import tensorflow as tf

import atom_Config
import AtomNet
import Data

random.seed(19960322)

config = atom_Config.Config()
data = Data.Data(config)
model = AtomNet.AtomNet(config)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True
with tf.Session(config=gpu_config) as sess:
	print('begin init...')
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
	print('finish init.')
	for e in range(config.epoch):
		step = 0
		right = []
		total = []
		loss = []
		begin_time = time.time()
		for item in data.get_train_data():
			feed_dict, _total = model.make_feed_dict(item)
			_, _loss, _right = sess.run([model.train_step, model.loss, model.right], feed_dict=feed_dict)
			right.append(_right)
			total.append(_total)
			loss.append(_loss)
			step += 1
			if step % 100 == 0:
				print('train---epoch: %d, step: %d, time: %f, loss: %f, precise: %f' % (e, step, (time.time() - begin_time) / 100, np.mean(loss), np.sum(right) / np.sum(total)))
				if step % 200 == 0 and step > 0:
					right = []
					total = []
					loss = []
					for item in data.get_dev_data():
						feed_dict, _total = model.make_feed_dict(item)
						_loss, _right = sess.run([model.loss, model.right], feed_dict=feed_dict)
						right.append(_right)
						total.append(_total)
						loss.append(_loss)
					print('dev---epoch: %d, step: %d, loss: %f, precise: %f' % (e, step, np.mean(loss), np.sum(right) / np.sum(total)))

					right = []
					total = []
					loss = []
					for item in data.get_test_data():
						feed_dict, _total = model.make_feed_dict(item)
						_loss, _right = sess.run([model.loss, model.right], feed_dict=feed_dict)
						right.append(_right)
						total.append(_total)
						loss.append(_loss)
					print('test---epoch: %d, step: %d, loss: %f, precise: %f' % (e, step, np.mean(loss), np.sum(right) / np.sum(total)))
				right = []
				total = []
				loss = []
				begin_time = time.time()

		right = []
		total = []
		loss = []
		for item in data.get_dev_data():
			feed_dict, _total = model.make_feed_dict(item)
			_loss, _right = sess.run([model.loss, model.right], feed_dict=feed_dict)
			right.append(_right)
			total.append(_total)
			loss.append(_loss)
		print('dev---epoch: %d, step: %d, loss: %f, precise: %f' % (e, step, np.mean(loss), np.sum(right) / np.sum(total)))

		right = []
		total = []
		loss = []
		for item in data.get_test_data():
			feed_dict, _total = model.make_feed_dict(item)
			_loss, _right = sess.run([model.loss, model.right], feed_dict=feed_dict)
			right.append(_right)
			total.append(_total)
			loss.append(_loss)
		print('test---epoch: %d, step: %d, loss: %f, precise: %f' % (e, step, np.mean(loss), np.sum(right) / np.sum(total)))

		saver.save(sess, os.path.join(config.checkpoints, 'model.cpkt'), global_step=e)
