import tensorflow as tf
from tensorflow.keras import Model, layers, backend
from tensorflow.keras.constraints import Constraint
from losses import disc_hinge, disc_loss, gen_loss, gen_hinge
from diff_augment import diff_augment
from tensorflow_addons.layers import SpectralNormalization

tf.random.set_seed(45)

class Generator(Model):
	def __init__(self, n_class=10, res=128):
		super(Generator, self).__init__()
		filters   = [  1024, 512, 256, 128,  64, 32]
		strides   = [     4,   2,   2,   2,   2,  2]
		self.cnn_depth  = len(filters)

		self.cond_embedding = layers.Embedding(input_dim=n_class, output_dim=50)
		self.cond_flat      = layers.Flatten()
		self.cond_dense     = layers.Dense(units=(8 * 8 * 1))
		self.cond_reshape   = layers.Reshape(target_shape=(64,))

		self.conv  = [SpectralNormalization(layers.Conv2DTranspose(\
					  filters=filters[idx], kernel_size=3,\
		              strides=strides[idx], padding='same',\
		              kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
					  use_bias=False))\
					  for idx in range(self.cnn_depth)]

		self.act   = [layers.LeakyReLU() for idx in range(self.cnn_depth)]

		self.bnorm = [layers.BatchNormalization() for idx in range(self.cnn_depth)]

		self.last_conv = SpectralNormalization(layers.Conv2D(filters=3, kernel_size=3,\
									   strides=1, padding='same',\
									   activation='tanh',\
									   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
									   use_bias=False))

	@tf.function
	def call(self, X):
		
		X = tf.expand_dims(tf.expand_dims(X, axis=1), axis=1)
		X = self.act[0]( self.conv[0]( X ) )

		for idx in range(1, self.cnn_depth):
			X = self.act[idx]( self.bnorm[idx]( self.conv[idx]( X ) ) )
		X = self.last_conv(X)
		return X


class Discriminator(Model):
	def __init__(self, n_class=10, res=128):
		super(Discriminator, self).__init__()
		filters    = [ 64, 128, 256, 512, 1024, 1]
		strides    = [  2,   2,   2,   2,    1, 1]
		self.cnn_depth = len(filters)

		self.cond_embedding = layers.Embedding(input_dim=n_class, output_dim=50)
		self.cond_flat      = layers.Flatten()
		self.cond_dense     = layers.Dense(units=(res * res * 1))
		self.cond_reshape   = layers.Reshape(target_shape=(res, res, 1))

		self.cnn_conv  = [layers.Conv2D(filters=filters[i], kernel_size=3,\
										strides=strides[i], padding='same',\
										kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
										use_bias=False)\
										for i in range(self.cnn_depth)] 

		self.cnn_bnorm = [layers.BatchNormalization() for _ in range(self.cnn_depth)]

		self.cnn_act   = [layers.LeakyReLU(alpha=0.2) for _ in range(self.cnn_depth)]

		self.flat      = layers.Flatten()

		self.disc_out  = layers.Dense(units=1)

	@tf.function
	def call(self, x, C):
		C = tf.expand_dims( tf.expand_dims(C, axis=1), axis=1)
		C = tf.tile(C, [1, x.shape[1], x.shape[2], 1])
		x = tf.concat([x, C], axis=-1)

		for layer_no in range(self.cnn_depth):
			x = self.cnn_act[layer_no]( self.cnn_bnorm[layer_no]( self.cnn_conv[layer_no]( x ) ) )
		reconst_x   = None
		x = self.disc_out( self.flat( x ) )

		return x, reconst_x

class DCGAN(Model):
	def __init__(self):
		super(DCGAN, self).__init__()
		self.gen    = Generator()
		self.disc   = Discriminator()

@tf.function
def dist_train_step(mirrored_strategy, model, model_gopt, model_copt, X, C, latent_dim=96, batch_size=64):

	diff_augment_policies = "color,translation"
	noise_vector          = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
	noise_vector_2        = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
	noise_vector          = tf.concat([noise_vector, C], axis=-1)
	noise_vector_2        = tf.concat([noise_vector_2, C], axis=-1)
	def train_step_disc(model, model_gopt, model_copt, X, C, latent_dim=96, batch_size=64):	
		with tf.GradientTape() as ctape:
			fake_img     = model.gen(noise_vector, training=False)

			X_aug        = diff_augment(X, policy=diff_augment_policies)
			fake_img     = diff_augment(fake_img, policy=diff_augment_policies)

			D_real, X_recon = model.disc(X_aug, C, training=True)
			D_fake, _       = model.disc(fake_img, C, training=True)

			c_loss       = disc_hinge(D_real, D_fake)

		variables = model.disc.trainable_variables
		gradients = ctape.gradient(c_loss, variables)
		model_copt.apply_gradients(zip(gradients, variables))
		return c_loss

	def train_step_gen(model, model_gopt, model_copt, X, C, latent_dim=96, batch_size=64):
		with tf.GradientTape() as gtape:
			
			fake_img_o   = model.gen(noise_vector, training=True)
			fake_img_2_o = model.gen(noise_vector_2, training=True)

			fake_img     = diff_augment(fake_img_o, policy=diff_augment_policies)
			fake_img_2   = diff_augment(fake_img_2_o, policy=diff_augment_policies)

			D_fake, _    = model.disc(fake_img, C, training=False)
			D_fake_2, _  = model.disc(fake_img_2, C, training=False)
			g_loss       = gen_hinge(D_fake) + gen_hinge(D_fake_2)
			mode_loss    = tf.divide(tf.reduce_mean(tf.abs(tf.subtract(fake_img_2_o, fake_img_o))),\
									tf.reduce_mean(tf.abs(tf.subtract(noise_vector_2, noise_vector)))
									)
			mode_loss   = tf.divide(1.0, mode_loss + 1e-5)
			g_loss      = g_loss + 1.0 * mode_loss

		variables = model.gen.trainable_variables
		gradients = gtape.gradient(g_loss, variables)
		model_gopt.apply_gradients(zip(gradients, variables))
		return g_loss

	per_replica_loss_disc = mirrored_strategy.run(train_step_disc, args=(model, model_gopt, model_copt, X, C, latent_dim, batch_size,))
	per_replica_loss_gen  = mirrored_strategy.run(train_step_gen, args=(model, model_gopt, model_copt, X, C, latent_dim, batch_size,))

	discriminator_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_disc, axis=None)
	generator_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_gen, axis=None)
	return generator_loss, discriminator_loss