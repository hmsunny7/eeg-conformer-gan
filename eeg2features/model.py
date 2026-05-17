import tensorflow as tf
from tensorflow.keras import Model, models, layers
import tensorflow_addons as tfa
import numpy as np

class MultiScaleCNN(layers.Layer):
    def __init__(self, filters=32, dropout=0.1):
        super().__init__()
        self.branch_s      = self._block(filters, k=3,  d=dropout)
        self.branch_m      = self._block(filters, k=7,  d=dropout)
        self.branch_l      = self._block(filters, k=13, d=dropout)
        self.res_proj      = layers.Dense(filters * 3, use_bias=False)
        self.bn_res        = layers.BatchNormalization()

    @staticmethod
    def _block(f, k, d):
        return models.Sequential([
            layers.Conv1D(f, k, padding='same', use_bias=False,
                          kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Activation('gelu'),
            layers.Dropout(d),
        ])

    def call(self, x, training=False):
        out = tf.concat([
            self.branch_s(x, training=training),
            self.branch_m(x, training=training),
            self.branch_l(x, training=training),
        ], axis=-1)
        res = self.bn_res(self.res_proj(x), training=training)
        return out + res

class ConvModule(layers.Layer):
    def __init__(self, units, kernel_size=15, dropout=0.1):
        super().__init__()
        self.norm        = layers.LayerNormalization(epsilon=1e-6)
        self.pw1         = layers.Conv1D(units * 2, 1)
        self.glu_gate    = layers.Lambda(
            lambda x: x[:, :, :units] * tf.sigmoid(x[:, :, units:])
        )
        self.dw_conv     = layers.DepthwiseConv1D(
            kernel_size, padding='same',
            depthwise_initializer='he_uniform',
            use_bias=False,
        )
        self.bn          = layers.BatchNormalization()
        self.act         = layers.Activation('swish')
        self.pw2         = layers.Conv1D(units, 1)
        self.drop        = layers.Dropout(dropout)

    def call(self, x, training=False):
        r = x
        x = self.norm(x)
        x = self.pw1(x)
        x = self.glu_gate(x)
        x = self.dw_conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x, training=training)
        return r + x


class ConformerBlock(layers.Layer):
    def __init__(self, units, num_heads=4, conv_kernel=15,
                 dropout=0.1, return_sequences=True):
        super().__init__()
        assert units % num_heads == 0
        self.return_sequences = return_sequences

        self.proj         = layers.Dense(units)
        self.pos_enc      = self._build_pos_enc(max_len=64, d=units)


        self.ffn1_norm    = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1         = models.Sequential([
            layers.Dense(units * 4, activation='swish'),
            layers.Dropout(dropout),
            layers.Dense(units),
            layers.Dropout(dropout),
        ])


        self.mha_norm     = layers.LayerNormalization(epsilon=1e-6)
        self.mha          = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=dropout,
        )
        self.mha_drop     = layers.Dropout(dropout)


        self.conv_module  = ConvModule(units, kernel_size=conv_kernel, dropout=dropout)


        self.ffn2_norm    = layers.LayerNormalization(epsilon=1e-6)
        self.ffn2         = models.Sequential([
            layers.Dense(units * 4, activation='swish'),
            layers.Dropout(dropout),
            layers.Dense(units),
            layers.Dropout(dropout),
        ])

        self.final_norm   = layers.LayerNormalization(epsilon=1e-6)
        self.gap          = layers.GlobalAveragePooling1D()

    @staticmethod
    def _build_pos_enc(max_len, d):
        pos   = np.arange(max_len)[:, None]
        i     = np.arange(d)[None, :]
        rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d))
        rads  = pos * rates
        enc   = np.zeros((max_len, d))
        enc[:, 0::2] = np.sin(rads[:, 0::2])
        enc[:, 1::2] = np.cos(rads[:, 1::2])
        return tf.constant(enc[None], dtype=tf.float32)

    def call(self, x, training=False):
        T = tf.shape(x)[1]

        x = self.proj(x) + tf.cast(self.pos_enc[:, :T, :], x.dtype)

        x = x + 0.5 * self.ffn1(self.ffn1_norm(x), training=training)

        xn = self.mha_norm(x)
        x  = x + self.mha_drop(
            self.mha(xn, xn, training=training), training=training
        )

        x = self.conv_module(x, training=training)

        x = x + 0.5 * self.ffn2(self.ffn2_norm(x), training=training)

        x = self.final_norm(x)

        if not self.return_sequences:
            x = self.gap(x)
            x = tf.math.l2_normalize(tf.cast(x, tf.float32), axis=-1)
        return x

class TripleNet(Model):
    def __init__(self, n_classes=10, n_features=128, dropout=0.5,
                 conv_kernel=15):
        super().__init__()

        self.cnn = MultiScaleCNN(filters=32, dropout=dropout)

        self.encoder = [
            ConformerBlock(
                units=32, num_heads=4,
                conv_kernel=conv_kernel,
                dropout=dropout,
                return_sequences=True,
            ),
            ConformerBlock(
                units=n_features, num_heads=4,
                conv_kernel=conv_kernel,
                dropout=dropout,
                return_sequences=False,
            ),
        ]

        self.n_features = n_features

    def call(self, x, training=False):
        x = self.cnn(x, training=training)
        for block in self.encoder:
            x = block(x, training=training)

        return x




@tf.function
def train_step(model, opt, X, Y):
    with tf.GradientTape() as tape:
        emb  = model(X, training=True)
        loss = tfa.losses.TripletSemiHardLoss()(Y, emb)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def test_step(model, X, Y):
    emb  = model(X, training=False)
    loss = tfa.losses.TripletSemiHardLoss()(Y, emb)
    return loss