import tensorflow as tf
import numpy as np
from tensorflow.keras import losses


def disc_hinge(dis_real, dis_fake):
	real_loss = -1.0 * tf.reduce_mean( tf.minimum(0.0, -1.0 + dis_real) )
	fake_loss = -1.0 * tf.reduce_mean( tf.minimum(0.0, -1.0 - dis_fake) )
	return (real_loss + fake_loss)/2.0

def gen_hinge(dis_fake):
	fake_loss = -1.0 * tf.reduce_mean( dis_fake )
	return fake_loss

def disc_loss(dis_real, dis_fake, dis_wrong=None):
	real = tf.convert_to_tensor(np.random.randint(low=7, high=12, size=dis_real.shape)/10.0)
	fake = tf.convert_to_tensor(np.random.randint(low=0, high=3, size=dis_real.shape)/10.0)
	real_loss  = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(real, dis_real)
	fake_loss  = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(fake, dis_fake)
	total_loss = (real_loss + fake_loss)/2.0
	return total_loss

def gen_loss(dis_fake):
	real = tf.ones_like(dis_fake)
	fake_loss = losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(real, dis_fake)
	return fake_loss

def critic_loss(D_real, D_fake):
	return  (tf.reduce_mean(D_fake) - tf.reduce_mean(D_real))

def wgan_gp_loss(D_real, D_fake, Y, Y_cap, model, batch_size):
	dloss = (tf.reduce_mean(D_fake) - tf.reduce_mean(D_real))
	lam   = 10
	eps   = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
	x_cap = eps * Y + (1-eps) * Y_cap
	with tf.GradientTape() as gptape:
		gptape.watch(x_cap)
		out = model.critic(x_cap, training=True)
	grad  = gptape.gradient(out, [x_cap])[0]
	grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
	grad_pen  = tf.reduce_mean((grad_norm - 1.0) ** 2)
	dloss     = dloss + lam * grad_pen
	return dloss