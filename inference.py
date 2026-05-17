import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from utils import load_complete_data, show_batch_images
from model import DCGAN, dist_train_step
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

for idx, item in enumerate(natsorted(glob('data/b2i_data/images/test/*')), start=0):
	clsname = os.path.basename(item)
	clstoidx[clsname] = idx
	idxtocls[idx] = clsname

image_paths = natsorted(glob('data/b2i_data/images/test/*/*'))
imgdict     = {}
for path in image_paths:
	key = path.split(os.path.sep)[-2]
	if key in imgdict:
		imgdict[key].append(path)
	else:
		imgdict[key] = [path]



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

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

	print(test_X.shape, test_Y.shape)

	test_path = []
	for X, Y in zip(test_X, test_Y):
		test_path.append(np.random.choice(imgdict[idxtocls[np.argmax(Y)]], size=(1,) ,replace=True)[0])

	test_batch  = load_complete_data(test_X, test_Y, test_path, batch_size=test_batch_size)
	X, Y, I      = next(iter(test_batch))

	print(X.shape, Y.shape, I.shape)

	gpus = tf.config.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync



	latent_dim = 128
	input_res  = 128





	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
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


	lr = 3e-4


	id = 25
	print('Generating test images for epoch: {}'.format(id))
	with mirrored_strategy.scope():
		model        = DCGAN()
		model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/best_ckpt', max_to_keep=30)
		ckpt.restore('experiments/ckpt/ckpt-{}'.format(id)).expect_partial()

	for cl in range(n_classes):
		test_noise  = np.random.uniform(size=(test_eeg_cls[cl].shape[0],128), low=-1, high=1)
		noise_lst   = np.concatenate([test_noise, test_eeg_cls[cl]], axis=-1)

		save_path = 'experiments/finalversion/{}/{}'.format(id, cl)
		if not os.path.isdir(save_path):
			os.makedirs(save_path)

		for idx, noise in enumerate(tqdm(noise_lst)):
			X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(noise, axis=0),))
			X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
			X = np.uint8(np.clip((X*0.5 + 0.5)*255.0, 0, 255))
			cv2.imwrite(save_path+'/{}_{}.jpg'.format(cl, idx), X)

	from thoughtviz_inceptionscore import inception_score
	inception_score(id)