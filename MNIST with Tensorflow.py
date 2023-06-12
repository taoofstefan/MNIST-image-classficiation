import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Model

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Building the model
class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')  # Convolutional layer with 32 filters and a 3x3 kernel
        self.flatten = Flatten()  # Flatten the input
        self.dense1 = Dense(128, activation='relu')  # Fully connected layer with 128 units
        self.dense2 = Dense(10, activation='softmax')  # Output layer with 10 units for classification
        
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

model = MNISTModel()

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()  # Loss function for multi-class classification
optimizer = tf.keras.optimizers.Adam()  # Adam optimizer for training the model

train_loss = tf.keras.metrics.Mean(name='train_loss')  # Metric to track the training loss
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # Metric to track the training accuracy

test_loss = tf.keras.metrics.Mean(name='test_loss')  # Metric to track the test loss
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')  # Metric to track the test accuracy

@tf.function
def train_step(inputs, outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_function(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(outputs, predictions)

@tf.function
def test_step(inputs, outputs):
    predictions = model(inputs)
    loss = loss_function(outputs, predictions)
    
    test_loss(loss)
    test_accuracy(outputs, predictions)

# Normalize the data by scaling it between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]  # Add a channel dimension to the input data
x_test = x_test[..., tf.newaxis]

# Create a training dataset by shuffling and batching the data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

# Create a test dataset by batching the data
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

epochs = 5

# Training loop
for epoch in range(epochs):
    # Iterate over the training dataset and perform a training step
    for train_inputs, train_labels in train_data:
        train_step(train_inputs, train_labels)
        
    # Iterate over the test dataset and perform a test step
    for test_inputs, test_labels in test_data:
        test_step(test_inputs, test_labels)
        
    # Print the current epoch's metrics
    template = 'Epoch {}, Train loss: {:.4f}, Train accuracy: {:.2f}%, Test loss: {:.4f}, Test accuracy: {:.2f}%'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100
    ))
    
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
