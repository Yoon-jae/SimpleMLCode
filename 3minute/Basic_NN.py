import tensorflow as tf
import numpy as np

x_data = np.array([
    [0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]
])
y_data = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([2, 3], -1., 1))
b = tf.Variable(tf.zeros([3]))

L = tf.nn.relu(tf.add(tf.matmul(X, W), b))
model = tf.nn.softmax(L)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})

        if step % 10 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)

    print("Predict : ", sess.run(prediction, feed_dict={X: x_data}))
    print("Target : ", sess.run(target, feed_dict={Y: y_data}))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))
    print("Accuracy : %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))