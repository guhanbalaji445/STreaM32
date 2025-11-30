# Hands-on Introduction to TensorFlow & Deep Learning

## 1. Introduction

Welcome to your hands-on experience with **TensorFlow** (TF). In this tutorial, we will explore the foundations of Deep Learning by solving computer vision problems.

We will cover:

1.  **The Basics**: Tensors and Data Loading.
2.  **Visualization**: How to look at your data before training.
3.  **Model 1 (Dense)**: Building a standard Multi-Layer Perceptron (MLP).
4.  **Model 2 (CNN)**: Building a Convolutional Neural Network (the gold standard for images).
5.  **Comparison**: Proving why CNNs are better for visual tasks.
6.  **Assignment**: An experimental challenge using a real-world dataset.

---

## 2. TensorFlow Basics: The "Tensor"

In TensorFlow, data is represented as **Tensors**. Think of a Tensor as a multi-dimensional array (like a NumPy array) that can run on a GPU.

* **Scalar (Rank 0):** `5`
* **Vector (Rank 1):** `[1, 2, 3]`
* **Matrix (Rank 2):** A 2D grid (e.g., a grayscale image).
* **Tensor (Rank 3+):** Higher dimensions (e.g., a color image with Height, Width, and Color Channels).

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")

# Creating a tensor
matrix = tf.constant([[1, 2], [3, 4]])
print(f"Shape: {matrix.shape}")
print(f"Rank: {tf.rank(matrix)}")
```

---

## 3. Project: Fashion Item Classification

We will build models to recognize articles of clothing (Sneakers, Shirts, Bags, etc.) using the **Fashion MNIST** dataset.

### Step 1: Load Data
```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training Data: {train_images.shape}") # (60000, 28, 28)
```

### Step 2: Preprocessing & Visualization
Before training, always inspect your data. Neural networks also need data normalized (0-1 range) rather than 0-255.

```python
# 1. Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2. Visualize a grid of images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

### Step 3: Model 1 - The Dense Network (MLP)
This is a standard "Feed Forward" network. It treats the image as a flat list of pixels, ignoring the spatial structure (shapes, edges).

```python
model_dense = tf.keras.Sequential([
    # Flatten: Turns 28x28 matrix into a single 784-long vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Dense: Fully connected layer. Learns global patterns.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Output: 10 neurons for 10 classes (Softmax = Probability)
    tf.keras.layers.Dense(10, activation='softmax')
])

model_dense.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train
history_dense = model_dense.fit(train_images, train_labels, epochs=5, validation_split=0.1)
```

### Step 4: Model 2 - The Convolutional Neural Network (CNN)
CNNs are designed for images. They use **filters** to scan the image for features (lines, curves) and **pooling** to reduce the size of the data while keeping important features.

**Crucial Step:** CNNs expect a 3D input `(Height, Width, Channels)`. Grayscale images only have 2 dimensions, so we must `reshape` them to add a channel dimension of 1.

```python
# Reshape for CNN: (60000, 28, 28, 1)
train_images_cnn = train_images.reshape(-1, 28, 28, 1)
test_images_cnn = test_images.reshape(-1, 28, 28, 1)

model_cnn = tf.keras.Sequential([
    # Conv2D: 32 filters, 3x3 kernel. Learns edges/textures.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # MaxPooling: Reduces size by half (28x28 -> 14x14). Keeps strongest features.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Another Conv block to learn complex patterns (shapes)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten and Classify
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train
history_cnn = model_cnn.fit(train_images_cnn, train_labels, epochs=5, validation_split=0.1)
```

### Step 5: Comparing Results
Let's plot the accuracy of both models to see which is better.

```python
plt.plot(history_dense.history['val_accuracy'], label='Dense Network')
plt.plot(history_cnn.history['val_accuracy'], label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```
*You should see the CNN achieving higher accuracy faster than the Dense network.*

---

## 4. Assignment: The "Intel Image" Optimization Lab

**Dataset:** [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
**Goal:** In the tutorial, we just compared Dense vs. CNN. In this assignment, you will act like a Deep Learning Researcher. You will experiment with hyperparameters, architecture changes, and regularization to find the best possible model.

### Part 0: Data Setup (Boilerplate)
Use this code to load the dataset (images are 150x150, much larger than Fashion MNIST!).

```python
# Update 'directory' to point to your unzipped dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/dataset/seg_train/seg_train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/dataset/seg_train/seg_train',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32)

# Normalize data [0-1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
```

### Part 1: The "Parameter Explosion" Experiment
Deep Learning is often about resource management. Let's see why Dense networks are bad for large images.

1.  **Calculate the Input Size**: The images are `150 x 150` pixels with `3` color channels. Calculate $150 \times 150 \times 3$. This is your input vector size.
2.  **Build a Dummy Dense Model**: Create a `Sequential` model with just a `Flatten` layer and **one** `Dense(128)` layer.
3.  **Check Summary**: Run `model.summary()`. Look at the number of "Trainable Params".
    * *Question:* How many millions of parameters does this simple layer have? Why might this be slow to train?

### Part 2: The CNN Baseline
Build a standard CNN similar to the one in the tutorial (2 Conv blocks + Flatten + Dense Output).
* **Train it for 10 epochs.**
* **Plot Training Loss vs. Validation Loss.**
* *Observation:* Does the Training Loss keep going down while Validation Loss goes up? If so, you are **Overfitting**.

### Part 3: The Optimization (Try "Different Stuff")
Now, try to beat your Baseline model. Choose **at least two** of the following experiments and document how the results changed.

* **Experiment A (Go Deeper):** Add a 3rd (and maybe 4th) Convolutional block (`Conv2D` + `MaxPooling2D`). Does accuracy improve, or does training just get slower?
* **Experiment B (Regularization):** Overfitting is a common problem with this dataset. Add a `tf.keras.layers.Dropout(0.5)` layer right before your final output layer.
    * *Hypothesis:* This should slightly lower training accuracy but **improve** validation accuracy. Did it work?
* **Experiment C (Kernel Size):** Change the filter size in your Conv2D layers from `(3, 3)` to `(5, 5)`. Does seeing "larger chunks" of the image help for nature scenes (Mountains, Forests)?

### Submission
Submit your code and a short answer(Create your own github repo, upload your files and submit the link) : **"Which experiment gave you the best Validation Accuracy, and why do you think it worked?"**
