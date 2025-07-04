import numpy as np
import tensorflow as tf
import time
from PIL import Image as PILImage
import matplotlib.pyplot as plt


# Compute the element-wise product of two vectors using Tensorflow
size = (50000,)
u = tf.random.uniform(size)
v = tf.random.uniform(size)

start_time = time.perf_counter()
tf.math.multiply(u, v)
end_time = time.perf_counter()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")


# Compute the element-wise product of two vectors using a for loop
size = (5000,)
pre_a = np.zeros(size)
pre_b = np.zeros(size)

a = np.random.uniform(pre_a)
b = np.random.uniform(pre_b)

start_time = time.perf_counter()
result = np.zeros(size)

for i in range(size[0]):
    result= a[i]*b[i]

end_time = time.perf_counter()
execution_time=end_time-start_time

print(f"Execution time: {execution_time} seconds")

##We can see that using Tensorflow the operation is faster around 4 x 10^-3 seconds.

# define 2x2 matrics A and B and compute AB+3B
A = tf.constant([[1,2],[3,4]])
B = tf.constant([[4,1],[5,2]])
#AB
AB = tf.matmul(A,B)
#3B
ThreeB = tf.scalar_mul(3, B)
#AB+3B
final = tf.add(AB, ThreeB)
print(np.array(final))

print(A[0,1])
##the value in first row and 2nd column of A is 2. we put 0 first for row, because 0 is the first index and represent row 1 and 1 for column which is 2nd index or column 2

# its convenient to convert tensor objects to NumPy arrays and vice versa
print(A.numpy()) # convert tensor to numpy array

A_r1 = A.numpy()[0] # first row of A as numpy array
A_r1 = tf.constant(A_r1) # convert numpy array to tensor

print(A_r1) # convert numpy array to tensor 

print(B.numpy())


## MNIST Dataset
# load the MNIST dataset, it contains 70,000 samples of handwritten digits. 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() # load dataset and split into training and testing sets

## Display the first image. Use Python and the `train_labels` to print a message stating the digit in the image.
#display first image
image1 = np.array(train_images[0])
plt.imshow(image1)
plt.axis('off')
plt.show()
#print the message
print(train_labels[0])
#it's 5


### Preparing the Data
'''Some Observations:
- there are 2^32 = 4,294,967,296  floating point numbers in a 32-bit system.  
- while in normalized 32-bit system there are 2^32-2^24 ~ 4.28 billion numbers.   
- decimal precision of a 32-bit system is approximately 7 decimal digits. (24 total bits * (log10(2)~ 0.301) ~ 7.22) example: 123.4567
- decimal precision of a 32-bit floating point number in the range $0$ to $1$ is also about 7 decimal digits. (because in ormalized form we still have 24 bits) example: 0.1234567  

Normalize the training and testing datasets to reshape the images to be arrays of size `(784,)`.'''

train_images = tf.reshape(train_images / 255.0, (-1, 784))
test_images = tf.reshape(test_images / 255.0, (-1, 784))

# Convert the training and testing labels to categorical data.
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
#print(train_labels.shape)
#print(test_labels.shape)


# Initializing and Training Our Model
##Defining a feedforward neural network using Tensorflow 
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# specify our loss function and *optimizer*
loss_function = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy']) # initialize model weights and compile model

# Apply the model to the training data 
pred_lables = model(train_images)
loss_function(train_labels, pred_lables)

# Now let's train our model:
model.fit(train_images,
          train_labels, 
          epochs=5, 
          batch_size=32) # train model on training data for 5 epochs with batch size of 32

# Load the Fashion MNIST dataset & Display the first 5 images and print their labels.

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


#display first 5 images and their lables
for i in range(5):
    image = np.array(train_images[i])
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Label: {train_labels[i]}")
    plt.show()

# Prepare the training and testing data: normalize and reshape the images and convert the labels to categorical data. Then define and train a model with $784\times 128\times 64\times 32\times 10$ architecture 
##Have the ReLU activation function in every layer except the last. For the final layer, use softmax for the activation function. Use stochastic gradient descent to train the model and mean squared error for its loss function. Train until your model has at least $90\%$ accuracy on the training data. 
train_images = tf.reshape(train_images/255.0,(-1, 784))
print(train_images.shape)

train_labels = tf.keras.utils.to_categorical(train_labels)
print(train_labels.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

loss_function = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy']) # initialize model weights and compile model
pred_lables = model(train_images)
loss_function(train_labels, pred_lables)
model.fit(train_images,
          train_labels, 
          epochs=300, 
          batch_size=64)
##the accuracy: 92.31%