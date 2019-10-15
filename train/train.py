import argparse
import tensorflow as tf
from tqdm import tqdm

# deepdrone imports
import models
import util


parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="cache.h5", help="path to load the cache")
parser.add_argument("--num-epochs", default=20, help="number of epochs")
parser.add_argument("--batch-size", default=8, help="batch size")
args = parser.parse_args()

loader = util.DataLoader(args.cache, input="images", output="odom")
num_samples = loader.get_num_samples()
batch_size = args.batch_size

network = models.cnn
model = network.create(num_output=loader.get_output_dim())
optimizer = tf.optimizers.Adam(1e-4)


def loss_function(y, y_hat):
    # define mean squared error loss
    return tf.reduce_mean((y - y_hat) ** 2)


@tf.function  # comment out for eager execution (if you want to debug)
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)  # forward pass
        loss = loss_function(y, y_hat)  # compute loss
    grads = tape.gradient(loss, model.variables)  # compute gradient
    optimizer.apply_gradients(zip(grads, model.variables))  # update model
    return loss


alpha = 0.95
running_loss = None
for epoch in range(args.num_epochs):
    pbar = tqdm(range(num_samples // batch_size))
    for iter in pbar:
        x, y = loader.get_batch(batch_size)  # grab a batch
        loss = train_step(x, y)  # perform a single forwards+backwards pass

        running_loss = (1-alpha)*loss.numpy() + alpha*running_loss if running_loss is not None else loss.numpy()
        pbar.set_description("Loss: {0:.5f}".format(running_loss))
