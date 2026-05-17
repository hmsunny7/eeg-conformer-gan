import tensorflow as tf
# from utils import vis, load_batch#, load_data
from utils import load_complete_data, show_batch_images
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
from model import DiffusionModel
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
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync

	latent_dim = 128
	input_res  = 128

	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='eeg2features/experiments/best_ckpt', max_to_keep=5000)
	triplenet_path = 'eeg2features/experiments/best_ckpt/ckpt-165'
	triplenet_ckpt.restore(triplenet_path)
	print('TripletNet restored from the latest checkpoint: {}'.format(triplenet_ckpt.step.numpy()))
	_, latent_Y = triplenet(X, training=False)

	print('Extracting test eeg features:')
	test_image_count = 50000
	
	test_eeg_cls      = {}
	for E, Y, X in tqdm(test_batch):
		Y = Y.numpy()[0]
		if Y not in test_eeg_cls:
			# print(E.shape)
			test_eeg_cls[Y] = [np.squeeze(triplenet(E, training=False)[1].numpy())]
		else:
			test_eeg_cls[Y].append(np.squeeze(triplenet(E, training=False)[1].numpy()))
	
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
	with mirrored_strategy.scope():
		model = DiffusionModel(img_size=input_res, in_channels=3, cond_dim=latent_dim, T=1000)
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model, optimizer=optimizer)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/best_ckpt', max_to_keep=300)
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	START         = int(ckpt.step.numpy())
	
	if ckpt_manager.latest_checkpoint:
		print('Restored from last checkpoint epoch: {0}'.format(START))


	for cl in range(n_classes):

		cond_vectors = test_eeg_cls[cl]

		N = cond_vectors.shape[0]
		per_cls_image = int(np.ceil((test_image_count // n_classes) / N))

		cond_vectors = np.expand_dims(cond_vectors, axis=1)
		cond_vectors = np.tile(cond_vectors, [1, per_cls_image, 1])
		cond_vectors = np.reshape(cond_vectors, [-1, latent_dim])

		save_path = f'experiments/finalversion/210/{cl}'
		os.makedirs(save_path, exist_ok=True)

		print(f"Generating class {cl} ...")

		for idx in tqdm(range(cond_vectors.shape[0])):

			cond = tf.expand_dims(cond_vectors[idx], axis=0)

			X = mirrored_strategy.run(
				model.sample,
				args=(cond, 30)
			)

			X = tf.squeeze(X)
			X = tf.clip_by_value(X, -1., 1.)
			X = (X * 0.5 + 0.5) * 255.
			X = tf.cast(X, tf.uint8).numpy()

			X = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)

			cv2.imwrite(f"{save_path}/{cl}_{idx}.jpg", X)