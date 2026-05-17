import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# from utils import vis, load_batch#, load_data
from utils import load_complete_data, show_batch_images
from model import DCGAN, dist_train_step#, train_step
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
from natsort import natsorted
import wandb
import numpy as np
import cv2
from eeg2features.model import TripleNet
import math
tf.random.set_seed(45)
np.random.seed(45)

clstoidx = {}
idxtocls = {}

for idx, item in enumerate(natsorted(glob('data/b2i_data/images/train/*')), start=0):
	clsname = os.path.basename(item)
	clstoidx[clsname] = idx
	idxtocls[idx] = clsname

image_paths = natsorted(glob('data/b2i_data/images/train/*/*'))
imgdict     = {}
for path in image_paths:
	key = path.split(os.path.sep)[-2]
	if key in imgdict:
		imgdict[key].append(path)
	else:
		imgdict[key] = [path]

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

if __name__ == '__main__':

	n_channels  = 14
	n_feat      = 128
	batch_size  = 128
	test_batch_size  = 1
	n_classes   = 10

	with open('data/b2i_data/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	train_path = []
	for X, Y in zip(train_X, train_Y):
		train_path.append(np.random.choice(imgdict[idxtocls[np.argmax(Y)]], size=(1,) ,replace=True)[0])

	test_path = []
	for X, Y in zip(test_X, test_Y):
		test_path.append(np.random.choice(imgdict[idxtocls[np.argmax(Y)]], size=(1,) ,replace=True)[0])

	train_batch = load_complete_data(train_X, train_Y, train_path, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, test_path, batch_size=test_batch_size)
	X, Y, I      = next(iter(train_batch))
	latent_label = Y[:16]
	print(X.shape, Y.shape, I.shape)

	gpus = tf.config.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync
	latent_dim = 128
	input_res  = 128

	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='eeg2features/experiments/best_ckpt_v7.1', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	print('TripletNet restored from the latest checkpoint: {}'.format(triplenet_ckpt.step.numpy()))
	latent_Y = triplenet(X, training=False)

	print('Extracting test eeg features:')
	test_image_count = 50000
	
	test_eeg_cls      = {}
	for E, Y, X in tqdm(test_batch):
		Y = Y.numpy()[0]
		if Y not in test_eeg_cls:
			test_eeg_cls[Y] = [np.squeeze(triplenet(E, training=False)[0].numpy())]
		else:
			test_eeg_cls[Y].append(np.squeeze(triplenet(E, training=False)[0].numpy()))
	
	for _ in range(n_classes):
		test_eeg_cls[_] = np.array(test_eeg_cls[_])
		print(test_eeg_cls[_].shape)

	for cl in range(n_classes):
		N = test_eeg_cls[cl].shape[0]
		per_cls_image = int(math.ceil((test_image_count//n_classes) / N))
		test_eeg_cls[cl] = np.expand_dims(test_eeg_cls[cl], axis=1)
		test_eeg_cls[cl] = np.tile(test_eeg_cls[cl], [1, per_cls_image, 1])
		test_eeg_cls[cl] = np.reshape(test_eeg_cls[cl], [-1, latent_dim])
		print(test_eeg_cls[cl].shape)

	lr = 3e-4
	with mirrored_strategy.scope():
		model        = DCGAN()
		model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/ckpt', max_to_keep=300)
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	START         = int(ckpt.step.numpy()) // len(train_batch) + 1
	EPOCHS        = 300
	model_freq    = 355
	t_visfreq     = 355
	latent        = tf.random.uniform(shape=(16, latent_dim), minval=-0.2, maxval=0.2)
	latent        = tf.concat([latent, latent_Y[:16]], axis=-1)
	print(latent_Y.shape, latent.shape)
	
	if ckpt_manager.latest_checkpoint:
		print('Restored from last checkpoint epoch: {0}'.format(START))

	if not os.path.isdir('experiments/results'):
		os.makedirs('experiments/results')

	for epoch in range(START, EPOCHS):
		t_gloss = tf.keras.metrics.Mean()
		t_closs = tf.keras.metrics.Mean()

		tq = tqdm(train_batch)
		for idx, (E, Y, X) in enumerate(tq, start=1):
			batch_size   = X.shape[0]
			C = triplenet(E, training=False)
			gloss, closs = dist_train_step(mirrored_strategy, model, model_gopt, model_copt, X, C, latent_dim, batch_size)
			gloss = tf.reduce_mean(gloss)
			closs = tf.reduce_mean(closs)
			t_gloss.update_state(gloss)
			t_closs.update_state(closs)
			ckpt.step.assign_add(1)
			if (idx%model_freq)==0:
				ckpt_manager.save()
			if (idx%t_visfreq)==0:
				X = mirrored_strategy.run(model.gen, args=(latent,))
				print(X.shape, latent_label.shape)
				show_batch_images(X, save_path='experiments/results/{}.png'.format(int(ckpt.step.numpy())), Y=latent_label)

			tq.set_description('E: {}, gl: {:0.3f}, cl: {:0.3f}'.format(epoch, t_gloss.result(), t_closs.result()))

		with open('experiments/log.txt', 'a') as file:
			file.write('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}\n'.format(epoch, t_gloss.result(), t_closs.result()))
		print('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}'.format(epoch, t_gloss.result(), t_closs.result()))


		if (epoch%10)==0:
			save_path = 'experiments/inception/{}'.format(epoch)

			if not os.path.isdir(save_path):
				os.makedirs(save_path)

			for cl in range(n_classes):
				test_noise  = np.random.uniform(size=(test_eeg_cls[cl].shape[0],128), low=-1, high=1)
				noise_lst   = np.concatenate([test_noise, test_eeg_cls[cl]], axis=-1)

				for idx, noise in enumerate(tqdm(noise_lst)):
					X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(noise, axis=0),))
					X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
					X = np.uint8(np.clip((X*0.5 + 0.5)*255.0, 0, 255))
					cv2.imwrite(save_path+'/{}_{}.jpg'.format(cl, idx), X)