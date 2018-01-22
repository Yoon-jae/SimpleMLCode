import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdadeltaOptimizer(0.001)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else :
        sess.run(tf.global_variables_initializer())

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([train_op, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            total_cost += cost_val

        print("Epoch : ", epoch + 1)
        print("Avg cost : %.3f" % (total_cost / total_batch))

    print("Optimization finished !")

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)

    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))

    acc, labels = sess.run([accuracy, model], feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
    print("Accuracy : ", acc)

    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2, 5, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%d' % np.argmax(labels[i]))
        subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap=plt.cm.gray_r)

    plt.show()
