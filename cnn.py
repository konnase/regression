import tensorflow as tf
import data

data_path = "./dataset"
dataset = data.Data(data_path)
print(dataset.labels)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 10

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 14])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 1])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([14, 10], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([10]), name='b1')

W2 = tf.Variable(tf.random_normal([10, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# and the weights connecting the hidden layer to the output layer
W3 = tf.Variable(tf.random_normal([10, 1], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([1]), name='b3')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
# hidden_out = tf.nn.relu(hidden_out)

hidden_out = tf.add(tf.matmul(hidden_out, W2), b2)
# hidden_out = tf.nn.relu(hidden_out)

y_out = tf.add(tf.matmul(hidden_out, W3), b3)
# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(y_out)

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                              + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(dataset.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = dataset.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch: {}, cost = {:.3f}".format((epoch + 1), avg_cost))
    print(sess.run(y, feed_dict={x: dataset.test_features, y: dataset.test_labels}))
