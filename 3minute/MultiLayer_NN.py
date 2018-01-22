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

W1 = tf.Variable(tf.random_normal([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_normal([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_normal([20, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([20]))
b3 = tf.Variable(tf.zeros([3]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

# No activation function at output layer normally.
model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdadeltaOptimizer(0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y:y_data})

        if (step + 1) % 10 == 0:
            print("Step : ", (step + 1), " Cost : ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    prediction = tf.argmax(model, axis=1)
    target = tf.argmax(Y, axis=1)

    print("Predict : ", sess.run(prediction, feed_dict={X: x_data}))
    print("Target : ", sess.run(target, feed_dict={Y: y_data}))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target), tf.float32))
    print("Accuracy : %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))