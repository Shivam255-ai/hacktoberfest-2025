import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

# ----------------------
# Load and preprocess MNIST
# ----------------------
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5   # normalize to [-1,1]
# Keep shape (28,28)
print("Training data shape:", X_train.shape)

# ----------------------
# Build Generator
# ----------------------
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(28*28, activation='tanh'))
    model.add(Reshape((28, 28)))
    return model

generator = build_generator()

# ----------------------
# Build Discriminator
# ----------------------
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()

# Compile discriminator (trainable=True here)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# ----------------------
# Build GAN (Generator + Discriminator)
# ----------------------
discriminator.trainable = False   # freeze only in GAN training
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# ----------------------
# Training Loop
# ----------------------
def train_gan(epochs=5000, batch_size=64, save_interval=1000):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # ---------------------
        # Train Discriminator
        # ---------------------
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_imgs = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        # Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))  # generator wants labels=1
        g_loss = gan.train_on_batch(noise, valid_y)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"{epoch+1} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Save generated images
        if (epoch + 1) % save_interval == 0:
            save_imgs(epoch+1)

# ----------------------
# Helper to save generated images
# ----------------------
def save_imgs(epoch):
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise, verbose=0)

    # Rescale images to [0,1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(5, 5, figsize=(5,5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.show()

# ----------------------
# Start training
# ----------------------
train_gan(epochs=2000, batch_size=64, save_interval=500)
