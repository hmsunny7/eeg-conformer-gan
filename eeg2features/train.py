import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

np.random.seed(45)
tf.random.set_seed(45)

if __name__ == '__main__':
	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 1
	n_classes   = 10

	with open('../data/b2i_data/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	print("Train Y:", train_Y[:10])
	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
	X, Y = next(iter(train_batch))

	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	START = int(triplenet_ckpt.step) // len(train_batch)
	if triplenet_ckptman.latest_checkpoint:
		print('Restored from the latest checkpoint, epoch: {}'.format(START))
	EPOCHS = 3000
	cfreq  = 10

	for epoch in range(START, EPOCHS):
		train_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
		train_loss = tf.keras.metrics.Mean()
		test_acc   = tf.keras.metrics.SparseCategoricalAccuracy()
		test_loss  = tf.keras.metrics.Mean()

		tq = tqdm(train_batch)
		for idx, (X, Y) in enumerate(tq, start=1):
			loss = train_step(triplenet, opt, X, Y)
			train_loss.update_state(loss)
			tq.set_description('Train Epoch: {}, Loss: {}'.format(epoch, train_loss.result()))

		tq = tqdm(val_batch)
		for idx, (X, Y) in enumerate(tq, start=1):
			loss = test_step(triplenet, X, Y)
			test_loss.update_state(loss)
			tq.set_description('Test Epoch: {}, Loss: {}'.format(epoch, test_loss.result()))

		triplenet_ckpt.step.assign_add(1)
		if (epoch%cfreq) == 0:
			triplenet_ckptman.save()