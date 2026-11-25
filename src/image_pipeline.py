import tensorflow as tf

def load_image(path, label, IMG_SIZE=(224, 224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    label = tf.one_hot(label, depth=7)
    return image, label

def create_dataset(df, training=False, BATCH_SIZE=32, IMG_SIZE=(224, 224)):
    paths = df["Path"].astype(str).values
    labels = df["dx_encoded"].astype("int32").values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=42
        )

    ds = ds.map(
        lambda path, label: load_image(path, label, IMG_SIZE),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds