import numpy as np
import tensorflow as tf
from model import Model
import time
import matplotlib.pyplot as plt
from utils import SMOTE, split_data, add_oversampling, compute_tfidf
from sklearn.datasets import make_classification
import pickle


class Config(object):
    batch_size = 64
    n_samples = 90795
    n_features = 88
    n_classes = 2
    max_epochs = 50
    lr = 1e-2
    l2 = 0.2
    add_data_weights = True
    hidden_size = 32
    embed_size = 50
    vocab_length = 9970
    dropout_prob = 0.1
    
    
class DeepNetModel(Model):
    """2-layer deep network.
    """
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_classes))
        self.bill_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.vocab_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        
    def create_feed_dict(self, input_batch, label_batch, bill_batch, dropout):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch,
            self.bill_placeholder: bill_batch,
            self.dropout_placeholder: dropout
        }
        return feed_dict

    def add_embedding(self, bills_batch):
        with tf.device("/cpu:0"):
            embeddings = tf.get_variable("Embedding",
                                         [self.config.vocab_length, self.config.embed_size],
                                         trainable=True)

            return tf.matmul(bills_batch, embeddings)
            
    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        
    def add_model(self, input_data):
        input_data = tf.nn.dropout(input_data, 1 - self.dropout_placeholder)

        h_weights = tf.Variable(tf.random_uniform([self.config.n_features + self.config.embed_size,
                                                  self.config.hidden_size]))
        h_biases = tf.Variable(tf.random_uniform([self.config.hidden_size]))
        
        h = tf.nn.sigmoid(tf.matmul(input_data, h_weights) + h_biases)
        h = tf.nn.dropout(h, 1 - self.dropout_placeholder)
        
        weights = tf.Variable(tf.ones([self.config.hidden_size, self.config.n_classes]))
        biases = tf.Variable(tf.ones([self.config.n_classes]))
        logits = tf.matmul(h, weights) + biases

        reg = self.config.l2 * (0.5 * tf.nn.l2_loss(weights) + 0.5 * tf.nn.l2_loss(h_weights))
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
        
    def run_epoch(self, sess, input_data, input_bills, input_labels):
        avg_loss = 0
        for step, (input_batch, bill_batch, label_batch) in enumerate(split_data(
                input_data, batch_size=self.config.batch_size,
                input_bills=input_bills, input_labels=input_labels)):
            feed_dict = self.create_feed_dict(input_batch, label_batch, bill_batch,
                                              self.config.dropout_prob)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            avg_loss += loss
        avg_loss = avg_loss / step
        return avg_loss
        
    def fit(self, sess, input_data, input_bills, input_labels, valid_data,
            valid_bills, valid_labels):
        losses = []
        train_accuracies = []
        valid_accuracies = []
        for epoch in range(self.config.max_epochs):
            start_t = time.time()
            avg_loss = self.run_epoch(sess, input_data, input_bills, input_labels)
            duration = time.time() - start_t
            print "Loss at epoch %d: %.2f (%.3f sec)" % (epoch, avg_loss, duration)
            tr_acc, tr_nays = self.test_accuracy(sess, input_data, input_bills, input_labels)
            print "Training accuracy %.5f, %d nays predicted" % (tr_acc, tr_nays)
            valid_acc, valid_nays = self.test_accuracy(sess, valid_data, valid_bills, valid_labels)
            print "Validation accuracy %.5f, %d nays predicted" % (valid_acc, valid_nays)
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
      
    def test_accuracy(self, sess, data, bills, labels):
        differences = []
        nays = []
        data = split_data(data, self.config.batch_size, bills, labels)
        for _, (x, b, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, label_batch=y, bill_batch=b, dropout=0)
            preds = sess.run(self.predictions, feed_dict=feed)
            nays.append((np.argmax(preds, 1) == 1).sum())
            diff = np.equal(np.argmax(preds, 1), np.argmax(y, 1))
            differences.extend(np.float32(diff))
        return np.mean(differences), int(np.sum(nays))
                  
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.bills = self.add_embedding(self.bill_placeholder)
        self.logits = self.add_model(tf.concat(1, [self.input_placeholder, self.bills]))
        self.loss = self.add_loss_op(self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.predictions = tf.nn.softmax(self.logits)
        
        
def test_deepnet(oversample=True):
    if oversample:
        train_features = np.load("../data/oversample_feats.npy")
        train_labels = np.load("../data/oversample_labels.npy")
        train_bills = np.load("../data/oversample_bills.npy")
    else:
        train_features = np.load("../data/train_feats.npy")
        train_labels = np.load("../data/train_labels.npy")
        train_bills = np.load("../data/train_words.npy")

    valid_features = np.load("../data/valid_feats.npy")
    valid_labels = np.load("../data/valid_labels.npy")
    valid_bills = np.load("../data/valid_words.npy")

    # first is yay second is nay
    train_labels = np.array([(x * 1, -(x - 1)) for x in train_labels])
    valid_labels = np.array([(x * 1, -(x - 1)) for x in valid_labels])
    
    train_bills = compute_tfidf(train_bills)
    valid_bills = compute_tfidf(valid_bills)

    config = Config()
    with tf.Graph().as_default():
        model = DeepNetModel(config)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        losses, tr_acc, valid_acc = model.fit(sess, train_features, train_bills, train_labels,
                                              valid_features, valid_bills, valid_labels)
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
        
        
def test_on_fake_data():
    config = Config()
    data = make_classification(config.batch_size * 1000, 88, 11, 11,
                               n_classes=2, weights=[0.95, 0.05], n_clusters_per_class=4)
    train_features = data[0][:57601]
    train_labels = np.array([(x * 1, -(x - 1)) for x in data[1][:57601]])
    valid_features = data[0][57601:]
    valid_labels = np.array([(x * 1, -(x - 1)) for x in data[1][57601:]])
    
    with tf.Graph().as_default():
        model = DeepNetModel(config)
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
