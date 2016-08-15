import numpy as np
import tensorflow as tf
from model import Model
import time
import matplotlib.pyplot as plt
from utils import SMOTE, split_rnn, add_oversampling, compute_tfidf


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
    window_length = 4
    
    
class RecurrentNetModel(Model):
    """2-layer recurrent network.
    """
        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=(None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, self.config.n_classes))
        self.bill_placeholder = tf.placeholder(
            tf.float32, shape=(None, self.config.vocab_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        
    def create_feed_dict(self, input_batch, label_batch, bill_batch, initial_state, dropout):
        feed_dict = {
            self.input_placeholder: input_batch,
            self.labels_placeholder: label_batch,
            self.bill_placeholder: bill_batch,
            self.initial_state: initial_state,
            self.dropout_placeholder: dropout
        }
        return feed_dict

    def add_embedding(self, bill_batch):
        embeddings = tf.get_variable("Embedding",
                                     [self.config.vocab_length, self.config.embed_size],
                                     trainable=True)

        return tf.matmul(bill_batch, embeddings)
            
    def add_projection(self, outputs):
        with tf.variable_scope("Projection"):
            proj_W = tf.get_variable(
                'projection_weights', [self.config.hidden_size, self.config.n_classes])
            proj_b = tf.get_variable(
                "projection_biases", [self.config.n_classes])
            logits = [tf.matmul(o, proj_W) + proj_b for o in outputs]
            tf.add_to_collection("total_loss", self.config.l2 * tf.nn.l2_loss(proj_W))
        return logits
            
    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
        
    def add_model(self, input_data, input_bills):
        data = [x for x in tf.split(0, self.config.window_length,
                                    tf.concat(1, [input_data, input_bills]))]
        data = [tf.nn.dropout(x, 1 - self.dropout_placeholder) for x in data]

        with tf.variable_scope("RNN") as scope:
            self.initial_state = tf.zeros(shape=[self.config.batch_size, self.config.hidden_size])
            state = self.initial_state
            outputs = []
            for tstep, current_input in enumerate(data):
                if tstep > 0:
                    scope.reuse_variables()
                H = tf.get_variable("RNN_H", [self.config.hidden_size, self.config.hidden_size])
                I = tf.get_variable("RNN_I", [self.config.n_features + self.config.embed_size,
                                              self.config.hidden_size])
                b = tf.get_variable("RNN_b", [self.config.hidden_size])
                state = tf.nn.sigmoid(tf.matmul(state, H) + tf.matmul(current_input, I) + b)
                outputs.append(state)
            H_reg = self.config.l2 * tf.nn.l2_loss(H)
            I_reg = self.config.l2 * tf.nn.l2_loss(I)
            tf.add_to_collection("total_loss", H_reg)
            tf.add_to_collection("total_loss", I_reg)
            self.final_state = state
        outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in outputs]
            
        return outputs
        
    def add_loss_op(self, logits):
        targets = [x for x in tf.split(0, len(logits), self.labels_placeholder)]
        
        if self.config.add_data_weights:
            weights = [tf.cast(tf.truediv(tf.reduce_sum(x, 0), tf.reduce_sum(x)), "float32")
                       for x in targets]
            logits = [logits[i] * weights[i] for i in range(len(weights))]

        ce_losses = tf.concat(0, [tf.nn.softmax_cross_entropy_with_logits(logits[i], targets[i])
                                  for i in range(len(logits))])
        tf.add_to_collection("total_loss", tf.reduce_mean(ce_losses))
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss
        
    def run_epoch(self, sess, input_data):
        losses = []
        state = self.initial_state.eval(session=sess)
        for step, (input_batch, word_batch, label_batch) in enumerate(input_data):
            
            feed_dict = self.create_feed_dict(input_batch, label_batch, word_batch,
                                              state, self.config.dropout_prob)

            _, loss, state = sess.run([self.train_op, self.loss, self.final_state],
                                      feed_dict=feed_dict)
            
            losses.append(loss)
        avg_loss = np.exp(np.mean(losses))
        return avg_loss
        
    def fit(self, sess, input_data, input_words, input_bills, input_labels,
            valid_data, valid_words, valid_bills, valid_labels):
        losses = []
        train_accuracies = []
        valid_accuracies = []
        train_batches = [batch for batch in
                         split_rnn(input_data, input_words, input_bills, self.config.batch_size,
                                   self.config.window_length, input_labels)]
        valid_batches = [batch for batch in
                         split_rnn(valid_data, valid_words, valid_bills, self.config.batch_size,
                                   self.config.window_length, input_labels)]
        for epoch in range(self.config.max_epochs):
            start_t = time.time()
            avg_loss = self.run_epoch(sess, train_batches)
            duration = time.time() - start_t
            print "Loss at epoch %d: %.2f (%.3f sec)" % (epoch, avg_loss, duration)
            tr_acc, tr_nays = self.test_accuracy(sess, train_batches)
            print "Training accuracy %.5f, %d nays predicted" % (tr_acc, tr_nays)
            valid_acc, valid_nays = self.test_accuracy(sess, valid_batches)
            print "Validation accuracy %.5f, %d nays predicted" % (valid_acc, valid_nays)
            losses.append(avg_loss)
            train_accuracies.append(tr_acc)
            valid_accuracies.append(valid_acc)
        return losses, train_accuracies, valid_accuracies
          
    def test_accuracy(self, sess, data):
        differences = []
        nays = []
        for step, (x, z, y) in enumerate(data):
            feed = self.create_feed_dict(x, y, z, self.initial_state.eval(session=sess), 0)
            preds = sess.run(self.predictions, feed_dict=feed)
            preds = np.vstack(preds)
            nays.append((np.argmax(preds, 1) == 1).sum())
            diff = np.equal(np.argmax(preds, 1), np.argmax(y, 1))
            differences.extend(np.float32(diff))
        return np.mean(differences), int(np.sum(nays))
                  
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.bills = self.add_embedding(self.bill_placeholder)
        self.outputs = self.add_model(self.input_placeholder, self.bills)
        self.projection = self.add_projection(self.outputs)
        self.predictions = [tf.nn.softmax(tf.cast(o, "float64")) for o in self.projection]
        self.loss = self.add_loss_op(self.projection)
        self.train_op = self.add_training_op(self.loss)
        
        
def test_rnn():
    train_features = np.load("../data/train_feats.npy")
    train_labels = np.load("../data/train_labels.npy")
    train_words = np.load("../data/train_words.npy")
    train_bills = np.load("../data/train_bills.npy")

    valid_features = np.load("../data/valid_feats.npy")
    valid_labels = np.load("../data/valid_labels.npy")
    valid_words = np.load("../data/valid_words.npy")
    valid_bills = np.load("../data/valid_bills.npy")

    # first is yay second is nay
    train_labels = np.array([(x * 1, -(x - 1)) for x in train_labels])
    valid_labels = np.array([(x * 1, -(x - 1)) for x in valid_labels])
    
    train_words = compute_tfidf(train_words)
    valid_words = compute_tfidf(valid_words)

    config = Config()
    with tf.Graph().as_default():
        model = RecurrentNetModel(config)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        losses, tr_acc, valid_acc = model.fit(sess, train_features, train_words,
                                              train_bills, train_labels,
                                              valid_features, valid_words,
                                              valid_bills, valid_labels)
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
