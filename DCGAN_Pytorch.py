import numpy as np
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt
import cv2
import glob
import os

# data insert
cnt = 0
arrList = []
print(arrList)
path_dir = "C:/Users/HakRyul/Desktop/result_images/1"
file_list = os.listdir(path_dir)
for file in file_list:
    img_name = path_dir + "/" + file
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # imgfile= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2_imshow(img)
    img = cv2.resize(img, (28, 28))
    input_size = 28
    output_size = 28
    bin_size = input_size // output_size
    img = (
        img.reshape((output_size, bin_size, output_size, bin_size, 1))
        .max(3)
        .max(1)
    )

    arrList.append(img)

    cnt += 1
    print("체크한 이미지: " + str(cnt))

    """
  if cnt == 1000:
    break
  """

arrList = np.array(arrList)

print(type(arrList))
print(arrList.shape)

latent_dim = 100  # 노이즈 벡터의 크기

img_rows, img_cols = 28, 28
img_channels = 1  # 흑백이므로 1차원
# (x_train, _), (_, _) = mnist.load_data()

x_train = arrList
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
x_train = x_train.astype("float32")
x_train /= 255  # 이미지 정규화 -1~1


def generator_model():
    dropout = 0.4
    depth = 256  # 64+64+64+64
    dim = 7

    model = Sequential()
    # In: 100
    # Out: dim x dim x depth
    model.add(Dense(dim * dim * depth, input_dim=latent_dim))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))
    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))

    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 2), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 4), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(int(depth / 8), 5, padding="same"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation("relu"))

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    model.add(Conv2DTranspose(1, 5, padding="same"))
    model.add(Activation("sigmoid"))

    return model


# (W−F+2P)/S+1
def discriminator_model():
    depth = 64
    dropout = 0.4
    input_shape = (img_rows, img_cols, img_channels)

    model = Sequential()
    # In: 28 x 28 x 1, depth = 1
    # Out: 14 x 14 x 1, depth=64
    model.add(
        Conv2D(depth, 5, strides=2, input_shape=input_shape, padding="same")
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth * 2, 5, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth * 4, 5, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    model.add(Conv2D(depth * 8, 5, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))

    # Out: 1-dim probability
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model


discriminator = discriminator_model()
discriminator.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(lr=0.0002, decay=6e-8),
    metrics=["accuracy"],
)

generator = generator_model()


def adversarial_model():
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(lr=0.0001, decay=3e-8),
        metrics=["accuracy"],
    )
    discriminator.trainable = True
    return model


adversarial = adversarial_model()


def plot_images(saveToFile=False, fake=True, samples=16, noise=None, epoch=0):
    filename = "mnist.png"
    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, latent_dim])
        else:
            filename = "mnist_%d.png" % epoch
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [img_rows, img_cols])
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    if saveToFile:
        plt.savefig(filename)
        plt.close("all")
    else:
        plt.show()


def train(train_epochs=2000, batch_size=256, save_interval=0):
    noise_input = None
    if save_interval > 0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_dim])
    for epoch in range(train_epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # select a random half of images
        images_train = x_train[
            np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :
        ]

        # sample noise and generate a batch of new images
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_dim])
        images_fake = generator.predict(noise)

        # train the discriminator (real classified as ones and generated as zeros)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        d_loss = discriminator.train_on_batch(x, y)

        # ---------------------
        #  Train Generator
        # ---------------------

        # train the generator (wants discriminator to mistake images as real)
        y = np.ones([batch_size, 1])
        a_loss = adversarial.train_on_batch(noise, y)

        log_msg = "%d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
        log_msg = "%s  [A loss: %f, acc: %f]" % (log_msg, a_loss[0], a_loss[1])
        print(log_msg)
        if save_interval > 0:
            if (epoch + 1) % save_interval == 0:
                plot_images(
                    saveToFile=True,
                    samples=noise_input.shape[0],
                    noise=noise_input,
                    epoch=(epoch + 1),
                )


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


timer = ElapsedTimer()
train(train_epochs=1000, batch_size=256, save_interval=100)
timer.elapsed_time()
plot_images(fake=True)
plot_images(fake=False, saveToFile=True)