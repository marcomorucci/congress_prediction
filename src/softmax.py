import numpy as np
import tensorflow as tf
from model import Model
import time
import matplotlib.pyplot as plt


def split_data(input_data, input_labels, batch_size, label_size):
    n = np.int64(input_data.shape[0])
    n_batches = np.int64(np.floor(n / float(batch_size)))
    for i in np.arange(n_batches):
        d = input_data[i * batch_size:(i + 1) * batch_size, :]
        l = input_labels[i * batch_size:(i + 1) * batch_size]
        yield d, l

        
class Config(object):
    batch_size = 64
    n_samples = 90795
    n_features = 88
    n_classes = 2
    max_epochs = 50
    lr = 1e-4
    l2 = 0.01
    add_data_weights = True

    
class SoftmaxModel(Model):
    """Softmax (Logistic) regression to test things out.
    """
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_classes))
        
    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch
        }
        return feed_dict
        
    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        
    def add_model(self, input_data):
        weights = tf.Variable(tf.zeros([self.config.n_features, self.config.n_classes]))
        biases = tf.Variable(tf.zeros([self.config.n_classes]))
        logits = tf.matmul(input_data, weights) + biases
        reg = 0.5 * self.config.l2 * tf.nn.l2_loss(weights)
        tf.add_to_collection("total_loss", reg)
        return logits
        
    def add_loss_op(self, logits):
        ratios = tf.truediv(tf.reduce_sum(self.labels_placeholder, 0),
                            tf.reduce_sum(self.labels_placeholder))
        if self.config.add_data_weights:
            weighted_logits = tf.mul(logits, ratios)
        else:
            weighted_logits = logits
        tf.add_to_collection("total_loss", tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            weighted_logits, self.labels_placeholder)))
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss
        
    def run_epoch(self, sess, input_data, input_labels):
        avg_loss = 0
        for step, (input_batch, label_batch) in enumerate(
            split_data(input_data, input_labels,
                       batch_size=self.config.batch_size,
                       label_size=self.config.n_classes)):
            feed_dict = self.create_feed_dict(input_batch, label_batch)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            avg_loss += loss
        avg_loss = avg_loss / step
        return avg_loss
        
    def fit(self, sess, input_data, input_labels, valid_data, valid_labels):
        losses = []
        train_accuracies = []
        valid_accuracies = []
        for epoch in range(self.config.max_epochs):
            start_t = time.time()
            avg_loss = self.run_epoch(sess, input_data, input_labels)
            duration = time.time() - start_t
            print "Loss at epoch %d: %.2f (%.3f sec)" % (epoch, avg_loss, duration)
            tr_acc = self.test_accuracy(sess, input_data, input_labels)
            print "Training accuracy %.5f" % tr_acc
            valid_acc = self.test_accuracy(sess, valid_data, valid_labels)
            print "Validation accuracy %.5f" % valid_acc
            losses.append(avg_loss)
            train_accuracies.append(tr_acc)
            valid_accuracies.append(valid_acc)
        return losses, train_accuracies, valid_accuracies
    
    def predict(self, sess, X, y):
        yhat = []
        data = split_data(X, y, self.config.batch_size, self.config.n_classes)
        for _, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, label_batch=y)
            preds = sess.run(self.predictions, feed_dict=feed)
            yhat.extend(preds)
        return yhat
      
    def accuracy(self, yhat, y):
        diff = np.equal(np.argmax(yhat, 1), np.argmax(y, 1))
        acc = np.mean(np.float32(diff))
        return acc

    def test_accuracy(self, sess, data, labels):
        accuracy = []
        data = split_data(data, labels, self.config.batch_size, self.config.n_classes)
        for _, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, label_batch=y)
            preds = sess.run(self.predictions, feed_dict=feed)
            acc = self.accuracy(preds, y)
            accuracy.append(acc)
        return np.mean(accuracy)
    
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.logits = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.predictions = tf.nn.softmax(self.logits)
        
        
def test_softmax():
    train_features = np.load("../data/train_feats.npy")
    train_labels = np.load("../data/train_labels.npy")
    valid_features = np.load("../data/valid_feats.npy")
    valid_labels = np.load("../data/valid_labels.npy")
    # first is yay second is nay
    train_labels = np.array([(x * 1, -(x - 1)) for x in train_labels])
    valid_labels = np.array([(x * 1, -(x - 1)) for x in valid_labels])
    config = Config()
    with tf.Graph().as_default():
        model = SoftmaxModel(config)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        losses, tr_acc, valid_acc = model.fit(sess, train_features, train_labels,
                                              valid_features, valid_labels)
        plt.figure("Accuracy")
        x = range(config.max_epochs)
        plt.plot(x, tr_acc, label="training set accuracy")
        plt.plot(x, valid_acc, label="validation set accuracy")
        plt.legend(loc="best")
        plt.title("Accuracy at each epoch")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        
        plt.figure("Loss")
        plt.plot(x, losses)
        plt.title("Loss at each epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()