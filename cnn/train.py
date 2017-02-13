import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import cPickle as pickle

# Parameters 
# ========================


# Data loading params
tf.flags.DEFINE_float("test_sample_pct", .1, "Percent of the training data to use for validation")
tf.flags.DEFINE_float("dev_sample_pct", .1, "Percent of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "/Users/haohanchen/Dropbox/NN-votes/haohan_cnn/bill112_embeded.pickle", "Data File")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "comma-seperated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
data = pd.read_pickle("/Users/haohanchen/Dropbox/NN-votes/haohan_cnn/vote_bill_112.pickle")
# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

print data.shape

x_text = data['summary']
print x_text.loc[x_text == ""] # No empty summary here

# Congressman level features
ivs_congman = ['tenure_at_beginning_of_term', 
					  'general_margin', 'general_percent', 'incumbent','party_lost_presidential_vote', 'party_lost_presidential_vote_new',
          			  'presdiential_margin_new_district', 'presidential_margin', 'presidential_vote', 'presidential_vote_new_district',
          			  'primary_margin', 'primary_percent']

x_congressman = np.array(data[ivs_congman])
# One-hot encoding for party and states
x_congressman_state = np.array(pd.get_dummies(data['state_ab'], prefix = "state"))
x_congressman_party = np.array(pd.get_dummies(data['party_char'], prefix = "party"))
x_congressman = np.concatenate((x_congressman, x_congressman_state, x_congressman_party), axis = 1)

# One-hot encode y's
y = np.array(pd.get_dummies(data['new_vote']))
print "Coding Rule for y: %s ==> %s" % (data['new_vote'][1], y[1])



# Build vocabulary
# Use TF module to automatically build dic and indexed text rep.
max_doc_len = max([len(x.split(" ")) for x in x_text])
print "Maximum document length is %d." % max_doc_len
vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len)
x_bill = np.array(list(vocab_processor.fit_transform(x_text)), dtype="int32")


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_pct * float(len(y)))
x_bill_train, x_bill_dev = x_bill[:dev_sample_index], x_bill[dev_sample_index:]
x_congressman_train, x_congressman_dev = x_congressman[:dev_sample_index], x_congressman[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_bill_train.shape[1],
            num_congressman_feat = x_congressman_train.shape[1], 
            num_neurons_cong_feat = 64, 
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_bill_batch, x_congressman_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_bill: x_bill_batch,
              cnn.input_x_congressman: x_congressman_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_bill_batch, x_congressman_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x_bill: x_bill_batch,
              cnn.input_x_congressman: x_congressman_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        def batch_iter(data, batch_size, num_epochs, shuffle=False):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]



        # Generate batches
        batches = batch_iter(
            list(zip(x_bill_train, x_congressman_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_bill_batch, x_congressman_batch, y_batch = zip(*batch)
            print x_bill_batch[1], x_congressman_batch[1], y_batch[1]
            
            train_step(x_bill_batch, x_congressman_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_bill_dev, x_congressman_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
