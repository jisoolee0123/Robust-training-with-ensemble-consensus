from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalAveragePooling2D, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise

class convnet(object):
    def __init__(self, config, x_shape, y_shape, mom2=0.1):

        self.x_shape = x_shape
        self.num_class = y_shape
        self.config = config

        loss = 'categorical_crossentropy'

        # model construction
        self.f = self.build_classifier()
        opt = Adam(lr=self.config.learning_rate, beta_1=0.9, beta_2=0.999)
        self.f.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        self.pred_f = K.function(inputs=[self.f.layers[0].input, K.learning_phase()],
                                 outputs=[self.f.layers[-1].output])

        # Adjust learning rate and betas for Adam optimizer
        mom1 = 0.9
        self.alpha_plan = [self.config.learning_rate] * self.config.n_epoch
        self.beta1_plan = [mom1] * self.config.n_epoch

        for i in range(self.config.epoch_decay_start, self.config.n_epoch):
            rd = float((i - self.config.n_epoch) / (self.config.epoch_decay_start - self.config.n_epoch))
            self.alpha_plan[i] = rd * self.config.learning_rate
            self.beta1_plan[i] = rd * mom1 + (1.-rd) * mom2

    def adjust_learning_rate(self, optimizer, epoch):
        K.set_value(optimizer.lr, self.alpha_plan[epoch])
        K.set_value(optimizer.beta_1, self.beta1_plan[epoch])

    def build_classifier(self):
        x_input = Input(shape=self.x_shape)
        kernel_init = 'he_normal'

        net = x_input
        net = GaussianNoise(stddev=0.15)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Dropout(rate=0.25)(net)

        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(256, (3, 3), activation=None, padding='same', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = MaxPooling2D((2, 2), padding='same')(net)
        net = Dropout(rate=0.25)(net)

        net = Conv2D(512, (3, 3), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(256, (3, 3), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = Conv2D(128, (3, 3), activation=None, padding='valid', kernel_initializer=kernel_init)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU(alpha=0.01)(net)
        net = GlobalAveragePooling2D()(net)
        net = Dense(units=self.num_class, activation='softmax', kernel_initializer=kernel_init)(net)

        return Model(x_input, net)













