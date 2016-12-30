import numpy as np
import tensorflow as tf
from model import Model
import time
import matplotlib.pyplot as plt
from utils import SMOTE, split_data, add_oversampling, compute_tfidf
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score
from utils import load_data


class Layer(object):
    def __init__(self, size=None, name=None, activation=None, loss=None):
        if size is not None:
            self.size = size
        else:
            size = 1
        if name is not None:
            self.name = name
        else:
            self.name = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = tf.nn.relu
        if loss is not None:
            self.loss = loss
        else:
            self.loss = tf.nn.l2_loss

            
class Config(object):
    batch_size = 64
    n_samples = 90795
    n_features = 187
    n_classes = 2
    max_epochs = 100
    lr = 1e-2
    l2 = 0.01
    add_data_weights = False
    layers = [Layer(64, "hidden3", tf.nn.relu, tf.nn.l2_loss),
              Layer(16, "hidden2", tf.nn.relu, tf.nn.l2_loss)]
    dropout_prob = 0.001
    verbose = True

    
class DeepNetModel(Model):
    """k-layer deep network.
    """
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(self.config.batch_size, self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        
    def create_feed_dict(self, input_batch, label_batch, dropout):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch,
            self.dropout_placeholder: dropout
        }
        return feed_dict
            
    def add_training_op(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def add_layer(self, input, size, name, activation=tf.nn.relu, loss=tf.nn.l2_loss):
        with tf.variable_scope(name):
            kernel_shape = [input.get_shape().as_list()[1], size]
            weights = tf.get_variable("weights", kernel_shape,
                                      initializer=tf.random_uniform_initializer())
            biases = tf.get_variable("biases", kernel_shape[1],
                                     initializer=tf.random_uniform_initializer())
            l = activation(tf.matmul(input, weights) + biases)
            l = tf.nn.dropout(l, 1 - self.dropout_placeholder)
            tf.add_to_collection("regularization", self.config.l2 * loss(weights))
            return l
        
    def add_model(self, input_data):
        input_data = tf.nn.dropout(input_data, 1 - self.dropout_placeholder)
        prev_l = input_data

        for l in self.config.layers:
            prev_l = self.add_layer(prev_l, l.size, l.name, l.activation, l.loss)

        with tf.variable_scope("output"):
            prev_shape = prev_l.get_shape().as_list()
            out_weights = tf.Variable(tf.random_uniform([prev_shape[1], self.config.n_classes]))
            out_biases = tf.Variable(tf.random_uniform([self.config.n_classes]))
            tf.add_to_collection("regularization", self.config.l2 * tf.nn.l2_loss(out_weights))
        
        reg = tf.add_n(tf.get_collection("regularization"))
        reg = tf.truediv(reg, float(len(self.config.layers) + 1))
        tf.add_to_collection("total_loss", reg)
        
        logits = tf.matmul(prev_l, out_weights) + out_biases

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
        
    def run_epoch(self, sess, input_data):
        avg_loss = 0
        for step, (input_batch, __, label_batch) in enumerate(input_data):
            feed_dict = self.create_feed_dict(input_batch, label_batch, self.config.dropout_prob)
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            avg_loss += loss
        avg_loss = avg_loss / step
        return avg_loss
        
    def fit(self, sess, input_data, input_labels, valid_input, valid_labels):
        train_data = [d for d in split_data(
            input_data, batch_size=self.config.batch_size, input_labels=input_labels)]
        valid_data = [d for d in split_data(
            valid_input, batch_size=self.config.batch_size, input_labels=valid_labels)]
        for epoch in range(self.config.max_epochs):
            start_t = time.time()
            avg_loss = self.run_epoch(sess, train_data)
            duration = time.time() - start_t
            if self.config.verbose:
                print "Loss at epoch %d: %.2f (%.3f sec)" % (epoch, avg_loss, duration)
            tr_acc, tr_nays, tr_fpr, tr_tpr, tr_auc = self.test_accuracy(sess, train_data)
            if self.config.verbose:
                print "Training accuracy %.5f, %d nays predicted, auc: %f" % (tr_acc, tr_nays, tr_auc)
            valid_acc, valid_nays, valid_fpr, valid_tpr, valid_auc = self.test_accuracy(sess, valid_data)
            if self.config.verbose:
                print "Validation accuracy %.5f, %d nays predicted, auc: %f" % (valid_acc, valid_nays, valid_auc)
            self.stats["loss"].append(avg_loss)
            self.stats["train_accuracy"].append(tr_acc)
            self.stats["train_tpr"].append(tr_tpr)
            self.stats["train_fpr"].append(tr_fpr)
            self.stats["train_auc"].append(tr_auc)
            self.stats["valid_accuracy"].append(valid_acc)
            self.stats["valid_tpr"].append(valid_tpr)
            self.stats["valid_fpr"].append(valid_fpr)
            self.stats["valid_auc"].append(valid_auc)
        return self
    
    def predict(self, sess, X, y):
        yhat = []
        data = split_data(X, y, self.config.batch_size, self.config.n_classes)
        for _, (x, __, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, label_batch=y)
            preds = sess.run(self.predictions, feed_dict=feed)
            yhat.extend(preds)
        return yhat
      
    def test_accuracy(self, sess, data):
        differences = []
        nays = []
        all_y = []
        all_preds = []
        for _, (x, __, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, label_batch=y, dropout=0.0)
            preds = sess.run(self.predictions, feed_dict=feed)
            nays.append((np.argmax(preds, 1) == 1).sum())
            diff = np.equal(np.argmax(preds, 1), np.argmax(y, 1))
            differences.extend(np.float32(diff))
            all_y.extend(y[:, 0])
            all_preds.extend(preds[:, 0])
        fpr, tpr, thresh = roc_curve(all_y, all_preds)
        auc = roc_auc_score(all_y, all_preds)
        return np.mean(differences), int(np.sum(nays)), fpr, tpr, auc
                  
    def __init__(self, config):
        self.config = config
        self.stats = {"loss": [],
                      "train_accuracy": [],
                      "train_tpr": [],
                      "train_fpr": [],
                      "train_auc": [],
                      "valid_accuracy": [],
                      "valid_tpr": [],
                      "valid_fpr": [],
                      "valid_auc": []}
        self.add_placeholders()
        self.logits = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.logits)
        self.train_op = self.add_training_op(self.loss)
        self.predictions = tf.nn.softmax(self.logits)
        
        
def test_deepnet(config=Config()):
    train_features, train_labels, valid_features, valid_labels, _, __ = load_data()

    with tf.Graph().as_default():
        model = DeepNetModel(config)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        model = model.fit(sess, train_features, train_labels, valid_features, valid_labels)
        return model
    

def plot_stats(model, config=Config()):
    print "Loss value: %.3f" % model.stats["loss"][-1]
    print "Training: accuracy = %.3f, AUC = %.3f" % (model.stats["train_accuracy"][-1],
                                                     model.stats["train_auc"][-1])
    print "Validation: accuracy = %.3f, AUC = %.3f" % (model.stats["valid_accuracy"][-1],
                                                       model.stats["valid_auc"][-1])
    
    plt.figure("Accuracy")
    x = range(config.max_epochs)
    plt.plot(x, model.stats["train_accuracy"], label="training set accuracy")
    plt.plot(x, model.stats["valid_accuracy"], label="validation set accuracy")
    plt.legend(loc="best")
    plt.title("Accuracy at each epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.figure("Loss")
    plt.plot(x, model.stats["loss"])
    plt.title("Loss at each epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.figure("AUC")
    plt.plot(x, model.stats["train_auc"], label="training set AUC")
    plt.plot(x, model.stats["valid_auc"], label="validation set AUC")
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.title("AUC at each epoch")
    plt.legend(loc="best")

    plt.figure("Train ROCs")
    for i in range(config.max_epochs):
        plt.plot(model.stats["train_fpr"][i], model.stats["train_tpr"][i])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Training Set ROC curve at each epoch")

    plt.figure("Valid ROCs")
    for i in range(config.max_epochs):
        plt.plot(model.stats["valid_fpr"][i], model.stats["valid_tpr"][i])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Validation Set ROC curve at each epoch")

    plt.figure("Final ROC")
    plt.plot(model.stats["train_fpr"][config.max_epochs - 1],
             model.stats["train_tpr"][config.max_epochs - 1],
             label="Training set")
    plt.plot(model.stats["valid_fpr"][config.max_epochs - 1],
             model.stats["valid_tpr"][config.max_epochs - 1],
             label="Validation set")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="best")
    plt.title("ROC curve after training")

    plt.show()

        
def test_on_fake_data():
    config = Config()
    data = make_classification(config.batch_size * 1000, config.n_features, 30, 11,
                               n_classes=2, weights=[0.5, 0.5], n_clusters_per_class=8)
    train_features = data[0][:57601, :]
    train_labels = np.array([(x * 1, -(x - 1)) for x in data[1][:57601]])
    valid_features = data[0][57601:, :]
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
