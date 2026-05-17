import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

if __name__ == '__main__':

	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 256
	n_classes   = 10

	with open('../data/b2i_data/eeg/image/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)


	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckpt.restore('experiments/best_ckpt/ckpt-117')

	tq = tqdm(test_batch)
	feat_X  = np.array([])
	feat_Y  = np.array([])
	for idx, (X, Y) in enumerate(tq, start=1):
		feat = triplenet(X, training=False)
		feat_X = np.concatenate((feat_X, feat.numpy()), axis=0) if feat_X.size else feat.numpy()
		feat_Y = np.concatenate((feat_Y, Y.numpy()), axis=0) if feat_Y.size else Y.numpy()

	print(feat_X.shape, feat_Y.shape)

	kmeans = KMeans(n_clusters=n_classes,random_state=45)
	kmeans.fit(feat_X)
	labels = kmeans.labels_
	kmeanacc = cluster_acc(feat_Y, labels)

	print('Accuracy score: {0:0.2f}'. format(kmeanacc))

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
	tsne_results = tsne.fit_transform(feat_X)
	df = pd.DataFrame()
	df['label'] = feat_Y
	df['x1'] = tsne_results[:, 0]
	df['x2'] = tsne_results[:, 1]

	df.to_csv('experiments/infer_triplet_embed2D.csv')	

	df = pd.read_csv('experiments/infer_triplet_embed2D.csv')

	plt.figure(figsize=(16,10))
	
	sns.scatterplot(
	    x="x1", y="x2",
	    data=df,
	    hue='label',
	    palette=sns.color_palette("hls", n_classes),
	    legend="full",
	    alpha=0.4
	)




































	plt.title('k-means accuracy: {}%'.format(kmeanacc*100))
	plt.savefig('experiments/embedding.png')
