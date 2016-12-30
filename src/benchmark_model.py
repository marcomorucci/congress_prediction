import tensorflow as tf
from utils import load_data
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from deep_net_topics import DeepNetModel, Config
import statsmodels.discrete.discrete_model as sm


def sklearn_logit():
    train_f, train_l, valid_f, valid_l, _, __ = load_data()
    model = LogisticRegression()
    model.fit(train_f, train_l[:, 0])
    train_pred = model.predict(train_f)
    valid_pred = model.predict(valid_f)
    train_acc = accuracy_score(train_l[:, 0], train_pred)
    valid_acc = accuracy_score(valid_l[:, 0], valid_pred)
    train_p = model.predict_proba(train_f)
    valid_p = model.predict_proba(valid_f)
    train_auc = roc_auc_score(train_l[:, 0], train_p[:, 1])
    valid_auc = roc_auc_score(valid_l[:, 0], valid_p[:, 1])
    
    print "Training: accuracy = %.3f, auc = %.3f" % (train_acc, train_auc)
    print "Validation: accuracy = %.3f, auc = %.3f" % (valid_acc, valid_auc)

    
def tf_softmax():
    config = Config()
    config.layers = []
    config.l2 = 0.0001
    train_f, train_l, valid_f, valid_l, _, __ = load_data()
    with tf.Graph().as_default():
        model = DeepNetModel(config)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        model = model.fit(sess, train_f, train_l, valid_f, valid_l)


def sm_logit():
    train_f, train_l, valid_f, valid_l, _, __ = load_data()
    model = sm.Logit(train_l[:, 0], train_f)
    model = model.fit_regularized()
    train_p = model.predict(train_f)
    valid_p = model.predict(valid_f)
    train_auc = roc_auc_score(train_l[:, 0], train_p)
    valid_auc = roc_auc_score(valid_l[:, 0], valid_p)

    print "Training: auc = %.3f" % (train_auc)
    print "Validation: auc = %.3f" % (valid_auc)
