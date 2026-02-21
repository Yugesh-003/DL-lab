import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# =========================
# 1. Load dataset
# =========================
(ds_train, ds_test), ds_info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

num_classes = ds_info.features["label"].num_classes
class_names = ds_info.features["label"].names

print("Classes:", class_names)


# =========================
# 2. Visualize samples
# =========================
plt.figure(figsize=(8,8))

for i, (img, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(class_names[label])
    plt.axis("off")

plt.show()


# =========================
# 3. Preprocess
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32

def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

train_ds = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = ds_test.map(preprocess).batch(BATCH_SIZE)


# =========================
# 4. Simple CNN Model
# =========================
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================
# 5. Train
# =========================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)


# =========================
# 6. Plot accuracy & loss
# =========================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()

plt.show()


# =========================
# 7. Prediction demo
# =========================
for images, labels in test_ds.take(1):
    preds = model.predict(images)

plt.figure(figsize=(8,8))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    pred = class_names[tf.argmax(preds[i])]
    true = class_names[labels[i]]
    plt.title(f"P:{pred} | T:{true}")
    plt.axis("off")

plt.show()