from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy
import numpy as np
import librosa
from random import shuffle

class RNNConfig(object):
    hidden_dim = 64
    dropout_keep_prob = 1
    learning_rate = 1e-3
    num_epochs = 500
    batch_size = 128
    print_per_batch = 20
    save_tb_per_batch = 10

class ASRRNN(object):
    def __init__(self, config, width, height, num_classes):  # 20,150,10
        self.config = config
        self.input_x = tf.placeholder(tf.float32, [None, width, height], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        input_x = tf.reshape(self.input_x, [-1, 1, width*height])

        cell = tf.nn.rnn_cell.GRUCell

        cells_fw = [cell(256) for _ in range(1)]
        cells_bw = [cell(256) for _ in range(1)]

        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
        rnn, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=input_x,
            dtype=tf.float32,
            scope="rnn_classification")
        
        fc = tf.layers.dense(rnn, self.config.hidden_dim, activation=tf.nn.relu, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)

        self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        self.logits = tf.reshape(self.logits, [-1, 10])
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return numpy.eye(num_classes)[labels_dense]

def read_files(files):
    labels = []
    features = []
    for ans, files in files.items():
        for file in files:
        	try:
        		wave, sr = librosa.load(file, mono=True)
        	except:
        		print(file + ' failed to load')
        		continue
        	label = dense_to_one_hot(ans, 10)
        	labels.append(label)
        	mfcc = librosa.feature.mfcc(wave, sr)
        	mfcc = np.pad(mfcc, ((0, 0), (0, 150 - len(mfcc[0]))), mode='constant', constant_values=0)
        	features.append(np.array(mfcc))
        	print('reading '+file)
    return np.array(features), np.array(labels)


def load_files(path='data/spoken_numbers_pcm_mai/'):
    files = os.listdir(path)
    cls_files = {}
    for wav in files:
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        cls_files.setdefault(ans, [])
        cls_files[ans].append(path + wav)
    train_files = {}
    valid_files = {}
    test_files = {}
    for ans, file_list in cls_files.items():
        shuffle(file_list)
        all_len = len(file_list)
        train_len = int(all_len * 0.7)
        valid_len = int(all_len * 0.2)
        test_len = all_len - train_len - valid_len
        train_files[ans] = file_list[0:train_len]
        valid_files[ans] = file_list[train_len:train_len + valid_len]
        test_files[ans] = file_list[all_len - test_len:all_len]
    return train_files, valid_files, test_files


def batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(rnn, x_batch, y_batch, keep_prob):
    feed_dict = {
        rnn.input_x: x_batch,
        rnn.input_y: y_batch,
        rnn.keep_prob: keep_prob
    }
    return feed_dict


def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value


def train(argv=None):
    train_files, valid_files, test_files = load_files()
    train_features, train_labels = read_files(train_files)
    train_features = mean_normalize(train_features)
    print('read train files down')
    valid_features, valid_labels = read_files(valid_files)
    valid_features = mean_normalize(valid_features)
    print('read valid files down')
    test_features, test_labels = read_files(test_files)
    test_features = mean_normalize(test_features)
    print('read test files down')

    width = 20  # mfcc features
    height = 150
    classes = 10  # digits

    config = RNNConfig
    rnn = ASRRNN(config, width, height, classes)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join('rnn_model_256_dim64_1_gru', 'model.ckpt')
    tensorboard_train_dir = 'tensorboard_rnn_256_dim64_1_gru/train'
    tensorboard_valid_dir = 'tensorboard_rnn_256_dim64_1_gru/valid'

    if not os.path.exists(tensorboard_train_dir):
        os.makedirs(tensorboard_train_dir)
    if not os.path.exists(tensorboard_valid_dir):
        os.makedirs(tensorboard_valid_dir)
    tf.summary.scalar("loss", rnn.loss)
    tf.summary.scalar("accuracy", rnn.acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

    total_batch = 0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_features, train_labels)
        for x_batch, y_batch in batch_train:
            total_batch += 1
            feed_dict = feed_data(rnn, x_batch, y_batch, config.dropout_keep_prob)
            session.run(rnn.optim, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                train_loss, train_accuracy = session.run([rnn.loss, rnn.acc], feed_dict=feed_dict)
                valid_loss, valid_accuracy = session.run([rnn.loss, rnn.acc], feed_dict={rnn.input_x: valid_features,
                                                                                         rnn.input_y: valid_labels,
                                                                                         rnn.keep_prob: config.dropout_keep_prob})
                print('Steps:' + str(total_batch))
                print(
                    'train_loss:' + str(train_loss) + ' train accuracy:' + str(train_accuracy) + '\tvalid_loss:' + str(
                        valid_loss) + ' valid accuracy:' + str(valid_accuracy))
            if total_batch % config.save_tb_per_batch == 0:
                train_s = session.run(merged_summary, feed_dict=feed_dict)
                train_writer.add_summary(train_s, total_batch)
                valid_s = session.run(merged_summary, feed_dict={rnn.input_x: valid_features, rnn.input_y: valid_labels,
                                                                 rnn.keep_prob: config.dropout_keep_prob})
                valid_writer.add_summary(valid_s, total_batch)

        saver.save(session, checkpoint_path, global_step=epoch)
    test_loss, test_accuracy = session.run([rnn.loss, rnn.acc],
                                           feed_dict={rnn.input_x: test_features, rnn.input_y: test_labels,
                                                      rnn.keep_prob: config.dropout_keep_prob})
    print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))


if __name__ == '__main__':
    train()
