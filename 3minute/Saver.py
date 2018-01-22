import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_normal([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_normal([20, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([3]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdadeltaOptimizer(0.01)
train_op = optimizer.minimize(cost, global_step=global_step)

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})

        print('Step : %d' % sess.run(global_step))
        print('Cost : %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)

    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)

    print("Predict : ", sess.run(prediction, feed_dict={X: x_data}))
    print("Target : ", sess.run(target, feed_dict={Y: y_data}))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))
    print("Accuracy : %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))