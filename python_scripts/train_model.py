import json
import os
import random
from time import time
import tensorflow as tf
import tensorflow_similarity as tfsim
import gc
import numpy as np
from markdown import Markdown
from matplotlib import pyplot as plt
from tensorboard.notebook import display
from tensorflow_similarity.samplers.file_samplers import load_image
from IPython.display import Markdown, display
from os import listdir
from os.path import isfile, join
# INFO and WARNING messages are not printed.
# This must be run before loading other modules.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



tfsim.utils.tf_cap_memory()
gc.collect()
tf.keras.backend.clear_session()
print("TensorFlow:", tf.__version__)
print("TensorFlow Similarity", tfsim.__version__)


def save_json_with_classes(class_map, class_map_from_label_to_val):
    print("Started writing dictionary to a file")
    with open("./class_map.txt", "w") as fp:
        json.dump(class_map, fp)  # encode dict into JSON
    with open("./class_map_from_label_to_val.txt", "w") as fp:
        json.dump(class_map_from_label_to_val, fp)  # encode dict into JSON
    print("Done writing dict into .txt file")


def load_train_dataset(train_folder):
    x_train = []
    x_train_img = []
    y_train = []
    class_i = 0
    class_map_from_val_to_label = {-1: "No match"}
    class_map_from_label_to_val = {"No match": -1}

    for folder in sorted(listdir(train_folder)):
        class_map_from_val_to_label[class_i] = folder
        class_map_from_label_to_val[folder] = class_i
        abs_folder = join(train_folder, folder)
        for f in listdir(abs_folder):
            abs_path = join(abs_folder, f)
            x_train.append(abs_path)
            x_train_img.append(load_image(abs_path))
            y_train.append(class_i)
        class_i = class_i + 1
    return x_train, x_train_img, y_train, class_map_from_val_to_label,class_map_from_label_to_val


def load_test_dataset(test_folder, class_map):
    x_test = []
    x_test_img = []
    y_test = []
    for folder in sorted(listdir(test_folder)):
        abs_folder = join(test_folder, folder)
        for f in listdir(abs_folder):
            abs_path = join(abs_folder, f)
            x_test.append(abs_path)
            x_test_img.append(load_image(abs_path))
            y_test.append(class_map[folder])
    return x_test, x_test_img, y_test


def define_callbacks(test_ds):
    log_dir = "logs/%d/" % (time())
    k = 5  # @param {type:"integer"}
    num_targets = 30  # @param {type:"integer"}
    num_queries = 60  # @param {type:"integer"}
    # Setup EvalCallback by splitting the test data into targets and queries.
    queries_x, queries_y = test_ds.get_slice(0, num_queries)
    targets_x, targets_y = test_ds.get_slice(num_queries, num_targets)
    tsc = tfsim.callbacks.EvalCallback(
        queries_x,
        queries_y,
        targets_x,
        targets_y,
        metrics=["f1"],
        k=k,
        tb_logdir=log_dir  # uncomment if you want to track in tensorboard
    )
    val_loss = tfsim.callbacks.EvalCallback(
        queries_x,
        queries_y,
        targets_x,
        targets_y,
        metrics=["binary_accuracy"],
        k=k,
        tb_logdir=log_dir  # uncomment if you want to track in tensorboard
    )
    tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return [val_loss, tsc, tbc]


def compute_index(model, index_size):
    # What is indexed
    index_x, index_y = test_ds.get_slice(0, index_size)
    index_data = tf.cast(index_x, dtype="int32")  # casted so it can displayed
    model.reset_index()
    model.index(index_x, index_y, data=index_data)


def test_5_examples(model, index_size, query_size):
    # what will be used as never seen before queries to test performance
    test_x, test_y = test_ds.get_slice(index_size, query_size)
    test_y = [int(c) for c in test_y]
    test_data = tf.cast(test_x, dtype="int32")
    num_examples = 5
    num_neigboors = 5
    idxs = random.sample(range(len(test_y)), num_examples)
    batch = tf.gather(test_x, idxs)
    nns = model.lookup(batch, k=num_neigboors)
    for bid, nn in zip(idxs, nns):
        # view results close by
        # if test_y[bid] in train_cls:
        # display(Markdown("**Known Class**"))
        # else:
        # display(Markdown("**Unknown Class**"))
        tfsim.visualization.viz_neigbors_imgs(test_data[bid], test_y[bid], nn, class_mapping=class_map_val_to_label, cmap="Greys")


train_folder = "./dataset_v3/train"
x_train, x_train_img, y_train, class_map_val_to_label, class_map_from_label_to_val = load_train_dataset(train_folder)

save_json_with_classes(class_map_val_to_label,class_map_from_label_to_val)

test_folder = "./dataset_v3/test"
x_test, x_test_img, y_test = load_test_dataset(test_folder, class_map_from_label_to_val)

training_classes = len(class_map_val_to_label) - 1

examples_per_class_per_batch = 6  # @param {type:"integer"}
train_cls = np.arange(0, training_classes).tolist()
print(train_cls)
classes_per_batch = len(class_map_val_to_label)-1
classes_per_batch_test = len(set(y_test))

train_ds = tfsim.samplers.MultiShotFileSampler(
    x_train,
    y_train,
    load_example_fn=load_image,
    examples_per_class_per_batch=examples_per_class_per_batch,
    classes_per_batch=classes_per_batch,
    class_list=train_cls,
    steps_per_epoch=100,
)

test_ds = tfsim.samplers.MultiShotMemorySampler(
    x_test,
    y_test,
    load_example_fn=load_image,
    total_examples_per_class=7,
    classes_per_batch=classes_per_batch_test,
)
print("TRAIN:")
print(train_ds.class_list)
print(train_ds.num_examples)
print(train_ds.example_shape)

print("TEST")
print(test_ds.class_list)
print(test_ds.num_examples)
print(test_ds.example_shape)

callbacks = define_callbacks(test_ds)

embedding_size = 128  # @param {type:"integer"}

# building model


model = tfsim.architectures.EfficientNetSim(
    train_ds.example_shape,
    embedding_size,
    variant="B3",
    trainable="partial",
    pooling="gem",  # Can change to use `gem` -> GeneralizedMeanPooling2D
    gem_p=1.0,  # Increase the contrast between activations in the feature map.
)
epochs = 10 # @param {type:"integer"}
LR = 0.00003  # @param {type:"number"}
gamma = 80  # @param {type:"integer"} # Loss hyper-parameter. 256 works well here.
steps_per_epoch = 100
val_steps = 40
distance = "cosine"
# init similarity loss
# loss = tfsim.losses.CircleLoss(gamma=gamma)
#loss = tfsim.losses.MultiSimilarityLoss(distance=distance) #LR = 0.00005
loss = tfsim.losses.ArcFaceLoss(num_classes=classes_per_batch, embedding_size=embedding_size, name="ArcFaceLoss")
# loss = tfsim.losses.TripletLoss(distance=distance)

# compiling and training
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=loss,
    distance=distance,
    search="nmslib",
)
history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=val_steps,
    callbacks=callbacks,
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title(f"Loss: {loss.name} - LR: {LR}")
plt.show()

index_size = 60
query_size = 60
compute_index(model, index_size)
test_5_examples(model, index_size, query_size)

calibration = model.calibrate(
    np.array(x_train_img),
    np.array(y_train),
    extra_metrics=["precision", "recall", "binary_accuracy"],
    verbose=1,
)

save_path = "./model_efficientNet"  # @param {type:"string"}
model.save(save_path, save_index=True, compression=False)
